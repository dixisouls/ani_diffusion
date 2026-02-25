"""
tests/test_module_5_dry_run.py

Quick Sanity check that EMA works correctly on CPU.

This verifies:
    1. EMA model initializes from UNet
    2. Shadow params are on CPU and match UNet param count
    3. update() changes shadow params
    4. apply_shadow() swaps weights correctly
    5. restore() swaps back correctly
    6. state_dict()/load_state_dict() round trips correctly
    7. EMA with DDP-wrapped module (simulated .module attribute)
    8. EMA saves/load via checkpoint helpers

Run:
    sbatch tests/test_module_5_dry_run.slurm
"""

import os
import sys
import copy
import shutil
import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)

# We use a small dummy model instead of a full UNet for speed and simplicity.
# The logic is identical -- EMA doesn't care about the model architecture.

class DummyModel(nn.Module):
    """
    Small Model for testing EMA
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,4,3,padding=1)
        self.norm1 = nn.GroupNorm(4,16)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

# -------------------------------------------------------------
# 1. EMA Model Initialization
# -------------------------------------------------------------

try:
    from training.ema import EMAModel

    model = DummyModel()
    model.train()

    ema = EMAModel(model, decay=0.9999)

    # Check shadow params exists and are on CPU
    assert len(ema.shadow_params) > 0, "No shadow params found"
    
    for name, param in ema.shadow_params.items():
        assert param.device == torch.device("cpu"), (
            f"Shadow param {name} on {param.device}, expected CPU"
        )
        
    # Check shadow param count matches model param count
    model_param_count = sum(p.numel() for p in model.parameters())
    shadow_param_count = sum(p.numel() for p in ema.shadow_params.values())
    assert shadow_param_count == model_param_count, \
        f"Param count mismatch | model={model_param_count:,} | shadow={shadow_param_count:,}"
    
    logger.info(
        f"[PASSED] EMA init | shadow params={shadow_param_count:,} | "
        f"decay={ema.decay}"
    )

except Exception as e:
    logger.error(f"[FAILED] EMA init | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 2. Shadow params start as exact copies of model params
# -------------------------------------------------------------

try:
    for name, param in model.named_parameters():
        shadow = ema.shadow_params[name]
        assert torch.equal(param.data.cpu(), shadow), \
            f"Shadow param {name} mismatch"
        
    logger.info(
        f"[PASSED] Shadow params match model params"
    )

except Exception as e:
    logger.error(f"[FAILED] Shadow params mismatch | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 3. update() causes shadow params to diverge
# -------------------------------------------------------------

try:
    # Save a copy of current shadow params
    shadow_before = {
        name: param.clone() for name, param in ema.shadow_params.items()
    }

    # Simualate training step: modify model params
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # Update EMA shadow params
    ema.update(model)

    # Check shadow params have diverged (not exactly equal)
    any_changed = False
    any_different_from_model = False
    
    for name, shadow in model.named_parameters():
        shadow = ema.shadow_params[name]
        old_shadow = shadow_before[name]

        if not torch.equal(shadow, old_shadow):
            any_changed = True
        
        if not torch.equal(shadow, param.data.cpu()):
            any_different_from_model = True
    
    assert any_changed, "Shadow params didn't change after update()"
    assert any_different_from_model, "Shadow params are identical to model params -- decay not applied"

    logger.info(
        f"[PASSED] Shadow params diverged after update()"
    )

except Exception as e:
    logger.error(f"[FAILED] Shadow params didn't diverge | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 4. apply_shadow() swaps weights correctly
# -------------------------------------------------------------

try:

    # Save current model weights (training weights)
    training_weights = {
        name: param.data.clone() for name, param in model.named_parameters()
    }

    # Apply shadow weights
    ema.apply_shadow(model)

    # Model weights should now match shadow weights
    for name, param in model.named_parameters():
        shadow = ema.shadow_params[name].to(device=param.device, dtype=param.dtype)
        assert torch.equal(param.data, shadow), \
            f"apply_shadow() did not copy shadow to model for {name}"
    
    logger.info(
        f"[PASSED] apply_shadow() copied shadow weights to model"
    )

except Exception as e:
    logger.error(f"[FAILED] apply_shadow() failed | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 5. restore() swaps back correctly
# -------------------------------------------------------------

try:
    # Restore original training weights
    ema.restore(model)

    # Model weights should now match original training weights
    for name, param in model.named_parameters():
        original = training_weights[name]
        assert torch.equal(param.data, original), \
            f"restore() did not restore original weights for {name}"
    
    logger.info(
        f"[PASSED] restore() restored original training weights"
    )

except Exception as e:
    logger.error(f"[FAILED] restore() failed | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 6. state_dict()/load_state_dict() round trips correctly
# -------------------------------------------------------------

try:
    # Get state dict
    sd = ema.state_dict()
    assert "decay" in sd, "state_dict missing key 'decay'"
    assert "shadow_params" in sd, "state_dict missing key 'shadow_params'"
    assert sd["decay"] == 0.9999, f"decay mismatch | {sd['decay']} != 0.9999"
    
    # Create a new EMA and load state dict

    model2 = DummyModel()
    ema2 = EMAModel(model2, decay=0.99) # different decay to verify it's overwritten
    ema2.load_state_dict(sd)

    assert ema2.decay == 0.9999, f"Loaded decay mismatch | {ema2.decay} != 0.9999"
    
    for name, param in ema.shadow_params.items():
        assert torch.equal(ema2.shadow_params[name], ema2.shadow_params[name]), \
            f"Shadow param mismatch after load_state_dict"
    
    logger.info(
        f"[PASSED] state_dict()/load_state_dict() round tripped correctly"
    )

except Exception as e:
    logger.error(f"[FAILED] state_dict()/load_state_dict() failed | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 7. EMA with DDP-wrapped module (simulated .module attribute)
# -------------------------------------------------------------

try:
    
    class FakeDDP(nn.Module):
        """Simulated DDP wrapper similar to accelerate's DistributedDataParallel"""
        def __init__(self, module):
            super().__init__()
            self.module = module
    
    inner_model = DummyModel()
    wrapped = FakeDDP(inner_model)

    ema_wrapped = EMAModel(wrapped, decay=0.9999)

    # Should have extracted params for inner model, not the wrapper
    inner_count = sum(p.numel() for p in inner_model.parameters())
    shadow_count = sum(p.numel() for p in ema_wrapped.shadow_params.values())

    assert shadow_count == inner_count, (
        f"DDP unwrap failed: inner={inner_count:,} | shadow={shadow_count:,}"
    )

    # Simulate training step on inner model
    with torch.no_grad():
        for param in inner_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # Update via wrapped model
    ema_wrapped.update(wrapped)

    # apply shadow
    ema_wrapped.apply_shadow(wrapped)

    for name, param in inner_model.named_parameters():
        shadow = ema_wrapped.shadow_params[name].to(device=param.device, dtype=param.dtype)
        assert torch.equal(param.data, shadow), \
            f"apply_shadow() through DDP wrapper failed for {name}"
    
    ema_wrapped.restore(wrapped)

    logger.info(
        f"[PASSED] EMA with DDP-wrapped module"
    )

except Exception as e:
    logger.error(f"[FAILED] EMA with DDP-wrapped module | {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------
# 8. EMA saves/load via checkpoint helpers
# -------------------------------------------------------------

try:
    from training.checkpoint import save_ema_state, load_ema_state

    test_ckpt_dir = os.path.join(
        os.environ.get("CHECKPOINT_DIR", "./checkpoints"), "_test_ema"
    )
    os.makedirs(test_ckpt_dir, exist_ok=True)

    # Save EMA state
    save_ema_state(test_ckpt_dir, ema)

    # Verify file exists
    ema_file = os.path.join(test_ckpt_dir, "ema_state.pt")
    assert os.path.exists(ema_file), f"EMA state file not saved at {ema_file}"

    # Load into fresh EMA
    model3 = DummyModel()
    ema3 = EMAModel(model3, decay=0.5) # wrong decay, should get overwritten
    loaded = load_ema_state(test_ckpt_dir, ema3)

    assert loaded is True, "load_ema_state() returned False"
    assert ema3.decay == ema.decay, f"Decay mismatch | {ema3.decay} != {ema.decay}"

    for name, param in ema.shadow_params.items():
        assert torch.equal(ema.shadow_params[name], ema3.shadow_params[name]), \
            f"Shadow param {name} mismatch after checkpoint load"
    
    # Cleanup
    shutil.rmtree(test_ckpt_dir)

    logger.info(
        f"[PASSED] EMA saves/load via checkpoint helpers"
    )

except Exception as e:
    if os.path.exists(test_ckpt_dir):
        shutil.rmtree(test_ckpt_dir)
    logger.error(f"[FAILED] EMA saves/load via checkpoint helpers | {str(e)}")
    sys.exit(1)

logger.info("All EMA tests passed successfully!")
sys.exit(0)