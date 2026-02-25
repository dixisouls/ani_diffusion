"""
tests/test_module_5_integration.py

Integration test: runs a short training job with EMA enabled,
verifies EMA weights diverge from training weights, checkpoints
include EMA state, and EMA can be restored from checkpoint.

This test:
    1. Runs 15 training steps with EMA update each step
    2. Verifies EMA shadow params diverge from initial copy
    3. Saves a checkpoint with EMA state
    4. Verifies EMA state file exists in checkpoint
    5. Loads checkpoint and verifies EMA state is restored
    6. Tests apply_shadow()/restore() with accelerate-wrapped model

Launched via:
    accelerate launch tests/test_module_5_integration.py

Run via:
    sbatch tests/test_module_5_integration.slurm
"""

import os
import sys
import shutil
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from diffusers.optimization import get_scheduler

from utils.logger import get_logger
from models.loader import load_models
from models.scheduler import build_train_scheduler
from data.dataset import build_dataloader
from training.loop import train_one_step
from training.ema import EMAModel
from training.checkpoint import save_checkpoint, load_checkpoint, load_ema_state, EMA_STATE_FILENAME


def main():
    
    # --- Setup ---
    mixed_precision = os.environ.get("MIXED_PRECISION", "fp16")
    accelerator = Accelerator(mixed_precision=mixed_precision)
    logger = get_logger("test_m5_integration", local_rank=accelerator.local_process_index)

    set_seed(42)

    logger.info("-"*50)
    logger.info(
        f"Processes: {accelerator.num_processes} | devices: {accelerator.device}"
    )
    logger.info("-"*50)

    # --- Test Checkpoint Directory ---
    test_checkpoint_dir = os.path.join(
        os.environ.get("CHECKPOINT_DIR", "./checkpoints"),"_ema_integration_test"
    )
    if accelerator.is_main_process:
        os.makedirs(test_checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # --- Load Models ---
    logger.info("Loading models...")
    vae, text_encoder, tokenizer, unet = load_models()

    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    vae.to(device=accelerator.device, dtype=weight_dtype)
    text_encoder.to(device=accelerator.device, dtype=weight_dtype)
    
    # --- Scheduler, Optimizer, LR Scheduler ---
    noise_scheduler = build_train_scheduler()
    optimizer = AdamW(
        unet.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=15,
    )

    # --- Wrap with Accelerate ---
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    # --- Initialize EMA (after Accelerate wrap)
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9999))
    ema_model = EMAModel(
        unet,
        decay=ema_decay,
    )

    # Save initial shadow tensors for comparison later
    initial_shadow = {
        name: param.clone() for name, param in ema_model.shadow_params.items()
    }

    # --- Dataloader ---
    dataloader = build_dataloader(
        samples_to_skip=0,
        max_samples=1600,
    )

    # ------------------------------------------------------------
    # Phase 1: Train for 15 steps with EMA updates
    # ------------------------------------------------------------
    logger.info("Phase 1: Training 15 steps with EMA updates")
    
    global_step = 0
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    samples_seen = 0
    best_loss = float("inf")

    unet.train()

    for batch in dataloader:
        if global_step >= 15:
            break
        
        batch = {
            k: v.to(accelerator.device) for k, v in batch.items()
        }

        loss = train_one_step(
            batch=batch,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
        )

        # EMA update
        ema_model.update(unet)

        global_step += 1
        samples_seen += batch_size * accelerator.num_processes

        if global_step % 5 == 0:
            logger.info(
                f"Step {global_step} | Loss: {loss:.4f}"
            )
    
    # ------------------------------------------------------------
    # Verify EMA shadow params have diverged from initial copy
    # ------------------------------------------------------------
    
    if accelerator.is_main_process:
        diverged_count = 0
        total_params = len(ema_model.shadow_params)
        
        for name, param in ema_model.shadow_params.items():
            if not torch.equal(param, initial_shadow[name]):
                diverged_count += 1
        
        assert diverged_count > 0, \
            "EMA shadow params did not diverge after 15 steps"
        
        logger.info(f"[PASSED] EMA diverged | {diverged_count}/{total_params} params diverged")
    
    accelerator.wait_for_everyone()

    # ------------------------------------------------------------
    # Verify apply_shadow()/restore() functionality
    # ------------------------------------------------------------
    
    if accelerator.is_main_process:
        base_model = unet.module if hasattr(unet, "module") else unet

        # Save training weights
        training_weights = {
            name: param.data.clone()
            for name, param in base_model.named_parameters()
        }

        # Apply shadow
        ema_model.apply_shadow(unet)

        # Model should now have shadow weights
        shadow_applied = True
        for name, param in base_model.named_parameters():
            if name in ema_model.shadow_params:
                shadow = ema_model.shadow_params[name].to(
                    device=param.device,
                    dtype=param.dtype,
                )
                if not torch.allclose(param.data, shadow, atol=1e-6):
                    shadow_applied = False
                    logger.error(f"apply_shadow() mistmach for param {name}")
                    break
        
        assert shadow_applied, "apply_shadow() did not correctly swap weights"
        logger.info("[PASSED] apply_shadow() works with accelerate-wrapped model")

        # Restore training weights
        ema_model.restore(unet)

        restore_correct = True

        for name, param in base_model.named_parameters():
            if not torch.equal(param.data, training_weights[name]):
                restore_correct = False
                logger.error(f"restore() mismatch for param {name}")
                break
        
        assert restore_correct, "restore() did not correctly restore training weights"
        logger.info("[PASSED] restore() works with accelerate-wrapped model")
        
    accelerator.wait_for_everyone()

    # ------------------------------------------------------------
    # Phase 2: Save checkpoint with EMA state
    # ------------------------------------------------------------
    
    logger.info("Phase 2: Saving checkpoint with EMA state")
    avg_loss = loss
    best_loss = save_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=test_checkpoint_dir,
        global_step=global_step,
        samples_seen=samples_seen,
        epoch=0,
        best_loss=best_loss,
        current_loss=avg_loss,
        ema_model=ema_model,
    )

    if accelerator.is_main_process:
        ema_file = os.path.join(
            test_checkpoint_dir, f"step_{global_step}", EMA_STATE_FILENAME
        )
        assert os.path.exists(ema_file), f"EMA state file not found: {ema_file}"
        logger.info(f"[PASSED] EMA state saved to {ema_file}")
    
    accelerator.wait_for_everyone()

    # ------------------------------------------------------------
    # Phase 3: Load checkpoint and verify EMA state is restored
    # ------------------------------------------------------------
    
    logger.info("Phase 3: Loading checkpoint and verifying EMA state")
    
    # Create a fresh EMA to simulate resume
    ema_fresh = EMAModel(
        unet,
        decay=0.5, # should get overwritten
    )

    metadata = load_checkpoint(
        accelerator,
        test_checkpoint_dir,
        ema_model=ema_fresh,
    )

    if accelerator.is_main_process:
        assert metadata is not None, "Checkpoint load returned None"
        assert metadata["global_step"] == global_step, "Global step mismatch"

        # EMA decay should be restored
        assert ema_fresh.decay == ema_decay, \
            f"EMA decay mismatch: {ema_fresh.decay} != {ema_decay}"
        
        # Shadow params should match
        for name, param in ema_model.shadow_params.items():
            assert name in ema_fresh.shadow_params, (
                f"Missing param {name} after checkpoint load"
            )
            assert torch.equal(param, ema_fresh.shadow_params[name]), (
                f"Shadow param {name} mismatch after checkpoint load"
            )
        
        logger.info("[PASSED] Checkpoint load and EMA state restoration")
    
    accelerator.wait_for_everyone()

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    
    if accelerator.is_main_process:
        shutil.rmtree(test_checkpoint_dir)
        logger.info("Test checkpoint directory cleaned up")
    
    logger.info("All tests passed!")

    accelerator.end_training()

if __name__ == "__main__":
    main()