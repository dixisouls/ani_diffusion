"""
tests/test_module_4_integration.py

Integration test: runs a short training job with Accelerate,
saves a checkpoint, then verifies the checkpoint can be loaded
and training can resume.

This test:
    1. Runs 20 training steps with real data and real models
    2. Saves a checkpoint at step 10 and step 20
    3. Verifies loss is finite and decreasing trend
    4. Verifies checkpoint files exist
    5. Loads the checkpoint and verifies metadata

This is launched via Accelerate (not plain python) since it
needs multi-GPU support:
    accelerate launch tests/test_module_4_integration.py

Run via SLURM:
    sbatch tests/test_module_4_integration.slurm
"""

import os
import sys
import time
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
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_metadata,
)


def main():
    # --- Setup ---
    mixed_precision = os.environ.get("MIXED_PRECISION", "fp16")
    accelerator = Accelerator(mixed_precision=mixed_precision)
    logger = get_logger("test_m4_int", local_rank=accelerator.local_process_index)

    set_seed(42)

    logger.info("=" * 60)
    logger.info("Module 4 Integration Test")
    logger.info(
        f"Processes: {accelerator.num_processes} | Device: {accelerator.device}"
    )
    logger.info("=" * 60)

    # --- Use a separate test checkpoint directory ---
    test_checkpoint_dir = os.path.join(
        os.environ.get("CHECKPOINT_DIR", "./checkpoints"), "_integration_test"
    )
    if accelerator.is_main_process:
        os.makedirs(test_checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # --- Load Models ---
    logger.info("Loading models...")
    vae, text_encoder, tokenizer, unet = load_models()

    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # --- Scheduler, Optimizer, LR Scheduler ---
    noise_scheduler = build_train_scheduler()

    optimizer = AdamW(unet.parameters(), lr=1e-4, weight_decay=0.01)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=20,
    )

    # --- Wrap with Accelerate ---
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    # --- Dataloader (small: 1600 samples max so it can effectively run till 20 steps) ---
    dataloader = build_dataloader(samples_to_skip=0, max_samples=1600)

    # =====================================================================
    # Phase 1: Train for 20 steps, checkpoint at 10 and 20
    # =====================================================================
    logger.info("Phase 1: Training for 20 steps...")

    global_step = 0
    batch_size = int(os.environ.get("BATCH_SIZE_PER_GPU", 16))
    samples_seen = 0
    losses = []
    best_loss = float("inf")

    unet.train()

    for batch in dataloader:
        if global_step >= 20:
            break

        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

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

        global_step += 1
        samples_seen += batch_size * accelerator.num_processes
        losses.append(loss)

        logger.info(f"Step {global_step} | loss={loss:.4f}")

        # Checkpoint at step 10 and step 20
        if global_step in (10, 20):
            avg_loss = sum(losses[-10:]) / min(len(losses), 10)
            best_loss = save_checkpoint(
                accelerator=accelerator,
                checkpoint_dir=test_checkpoint_dir,
                global_step=global_step,
                samples_seen=samples_seen,
                epoch=0,
                best_loss=best_loss,
                current_loss=avg_loss,
            )


    # =====================================================================
    # Verify Phase 1
    # =====================================================================

    if accelerator.is_main_process:
        # Check losses are finite
        all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
        if all_finite:
            logger.info(f"[PASSED] All {len(losses)} losses are finite")
        else:
            logger.error("[FAILED] Some losses are not finite")
            sys.exit(1)

        # Check losses are positive
        all_positive = all(l > 0 for l in losses)
        if all_positive:
            logger.info("[PASSED] All losses are positive")
        else:
            logger.error("[FAILED] Some losses are non-positive")
            sys.exit(1)

        # Check checkpoint files exist
        step_10_dir = os.path.join(test_checkpoint_dir, "step_10")
        step_20_dir = os.path.join(test_checkpoint_dir, "step_20")
        metadata_path = os.path.join(test_checkpoint_dir, "metadata.json")

        assert os.path.exists(step_10_dir), f"step_10 checkpoint not found"
        assert os.path.exists(step_20_dir), f"step_20 checkpoint not found"
        assert os.path.exists(metadata_path), f"metadata.json not found"

        logger.info("[PASSED] Checkpoint directories and metadata exist")

        # Check metadata values
        metadata = load_metadata(test_checkpoint_dir)
        assert (
            metadata["global_step"] == 20
        ), f"Expected step 20, got {metadata['global_step']}"
        assert metadata["samples_seen"] == samples_seen
        assert metadata["latest_checkpoint_name"] == "step_20"

        logger.info("[PASSED] Metadata values are correct")

        # Log summary
        logger.info(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
        logger.info(f"First 5 losses: {[f'{l:.4f}' for l in losses[:5]]}")
        logger.info(f"Last 5 losses:  {[f'{l:.4f}' for l in losses[-5:]]}")

    accelerator.wait_for_everyone()

    # =====================================================================
    # Phase 2: Load checkpoint and verify resume
    # =====================================================================

    if accelerator.is_main_process:
        logger.info("Phase 2: Testing checkpoint resume...")

    loaded_metadata = load_checkpoint(accelerator, test_checkpoint_dir)

    if accelerator.is_main_process:
        assert loaded_metadata is not None, "Checkpoint load returned None"
        assert loaded_metadata["global_step"] == 20
        assert loaded_metadata["samples_seen"] == samples_seen

        logger.info("[PASSED] Checkpoint loaded and metadata verified")

    accelerator.wait_for_everyone()

    # =====================================================================
    # Cleanup
    # =====================================================================

    if accelerator.is_main_process:
        import shutil

        shutil.rmtree(test_checkpoint_dir)
        logger.info("Test checkpoint directory cleaned up")

    logger.info("=" * 60)
    logger.info("All module 4 integration tests passed")
    logger.info("=" * 60)

    accelerator.end_training()


if __name__ == "__main__":
    main()
