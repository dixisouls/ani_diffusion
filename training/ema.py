"""
training/ema.py

Exponential Moving Average (EMA) for UNet weights.

EMA maintains a shadow copy of model parameters that is smoothed
average over training steps.

Update rule (every step):
    ema_param = decay * ema_param + (1 - decay) * current_param

- Shadow params stored on CPU to save GPU memory.
- Custom implementation (diffusers.EMAModel can interfere with accelerate's DDP)
to get clean interactions with accelerate's DDP.
- State dict save/load is seperate from accelerate's save_state
- apply_shadow()/restore() pattern for swapping weights at inference

Usage:
    from training.ema import EMAModel

    ema = EMAModel(unet, decay=0.9999)

    # in training
    ema.update(unet)

    # for inference/evaluation
    ema.apply_shadow(unet) # swap ema weights to model
    # run inference
    ema.restore(unet) # swap back to original weights

    # For checkpoints
    ema.state_dict() # return shadow params dict
    ema.load_state_dict(sd) # restores from checkpoint
"""

import copy
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

class EMAModel:
    """
    Exponential Moving Average for model parameters.

    Args:
        model: The model whose parameters to track (UNet).
               Can be a plain module or Accelerate-wrapped (DDP) module.
        decay: EMA decay rate. Higher = smoother, slower adaptation.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):

        self.decay = decay

        # Extract the underlying model if wrapped by Accelerate/DDP
        # Accelerate wraps the models in DistributedDataParallel which adds
        # .module attribute. We need the raw parameters

        base_model = model.module if hasattr(model, "module") else model

        # Deep copy all parameters to CPU as shadow weights
        # These are detached from the computation graph -- no gradients
        self.shadow_params = {
            name: param.data.clone().detach().cpu()
            for name, param in base_model.named_parameters()
        }

        # Backup storage for restore() after apply_shadow()
        self._backup_params = {}

        param_count = sum(p.numel() for p in self.shadow_params.values())
        logger.info(
            f"EMA intialized | decay={decay} | "
            f"shadow params={param_count:,} | device=cpu"
        )
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update shadow parameters with current model parameters.
        Called once per training step, after optimizer.step().

        Update:
            shadow = decay * shadow + (1 - decay) * current
        Args:
            model: The current model (UNet). Can be DDP-wrapped.
        """
        base_model = model.module if hasattr(model, "module") else model

        for name, param in base_model.named_parameters():
            if name in self.shadow_params:
                # Move current pararm to CPU for the update
                current = param.data.detach().cpu()
                self.shadow_params[name].mul_(self.decay).add_(
                    current, alpha=(1.0 - self.decay)
                )
    
    def apply_shadow(self, model: torch.nn.Module):
        """
        Swap model parameters with shadow parameters.
        
        Backs up the current (training) parameters so they can be restored
        later with restore(). Use this before inference/evaluation.

        Args:
            model: The model (UNet) to apply shadow weights to.
        """
        base_model = model.module if hasattr(model, "module") else model

        self._backup_params = {}

        for name, param in base_model.named_parameters():
            if name in self.shadow_params:
                # Backup current training weights
                self._backup_params[name] = param.data.clone()
                # Copy the shadow weights to model (on correct device and dtype)
                param.data.copy_(
                    self.shadow_params[name].to(
                        device=param.device,
                        dtype=param.dtype
                    )
                )
        
        logger.info(f"EMA shadow weights applied to model")
    
    def restore(self, model: torch.nn.Module):
        """
        Restore original model parameters after apply_shadow().

        Must be called after apply_shadow() before resume training,
        otherwise the model will train from the EMA weights.

        Args:
            model: The model (UNet) to restore original weights to.
        """
        if not self._backup_params:
            logger.warning("restore() called without prior apply_shadow() -- no-op")
            return
        
        base_model = model.module if hasattr(model, "module") else model

        for name, param in base_model.named_parameters():
            if name in self._backup_params:
                # Restore the backup weights
                param.data.copy_(self._backup_params[name])
        
        self._backup_params = {}
        logger.info(f"EMA weights restored to original training weights")
    
    def state_dict(self) -> dict:
        """
        Return EMA state for checkpointing.

        Returns:
            dict: with decay and shadow params
        """
        return {
            "decay": self.decay,
            "shadow_params": {
                name: param.clone() for name, param in self.shadow_params.items()
            }
        }
    
    def load_state_dict(self, state_dict: dict):
        """
        Load EMA state from checkpoint.

        Args:
            state_dict: dict with decay and shadow params
        """
        self.decay = state_dict["decay"]
        self.shadow_params = {
            name: param.clone().detach().cpu() 
            for name, param in state_dict["shadow_params"].items()
        }
        param_count = sum(p.numel() for p in self.shadow_params.values())
        logger.info(
            f"EMA state loaded | decay={self.decay} | "
            f"shadow params={param_count:,} | device=cpu"
        )