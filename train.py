"""
train.py

Main training orchestrator.

This the entry point launched via:
    accelerate launch train.py

It wires together all modules in order:
    1. Initialize Accelerate
    2. Load models (VAE, CLIP, UNet)
    3. Build noise scheduler
    4. Build optimizers and LR scheduler
    5. Build dataloader
    6. Wrap everything with Accelerate
    7. Check for existing checkpoint and resume if found
    8. Initialize WandB (only rank 0)
    9. Enter training loop
    10. Save checkpoints at regular intervals

No training logic here -- training/loop.py handles that.
No checkpoint management here -- training/checkpoint.py handles that.
"""

import os
import sys
import math
import time
import wandb
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
from training.checkpoint import save_checkpoint, load_checkpoint


def main():
    # -------------------------------------------------------------
    # 1. Initialize Accelerate
    # -------------------------------------------------------------
    mixed_precision = os.environ.get("MIXED_PRECISION", "fp16")

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        log_with=None,  # We handle WandB manually for more control
    )

    # Set up logger with correct rank
    logger = get_logger("train", local_rank=accelerator.local_process_index)

    # Reproducibility
    set_seed(42)

    logger.info("--------------------------------")
    logger.info("Starting Training")
    logger.info(f"Process Rank: {accelerator.local_process_index}")
    logger.info(f"Number of Processes: {accelerator.num_processes}")
    logger.info(f"Mixed Precision: {mixed_precision}")
    logger.info(f"Device: {accelerator.device}")
    logger.info("--------------------------------")

    # -------------------------------------------------------------
    # 2. Load Models
    # -------------------------------------------------------------
    logger.info("Loading Models...")
    vae, text_encoder, tokenizer, unet = load_models()

    # Move frozen models to device and set dtype for memory efficiency
    # VAE and text encoder are frozen -- cast to float16 to save memory
    # UNet stays in float32; Accelerate will handle mixed precision for training
    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    logger.info(
        f"Frozen models moved to device={accelerator.device} | dtype={weight_dtype}"
    )

    # -------------------------------------------------------------
    # 3. Build Noise Scheduler
    # -------------------------------------------------------------
    noise_scheduler = build_train_scheduler()

    # -------------------------------------------------------------
    # 4. Build Optimizers and LR Scheduler
    # -------------------------------------------------------------
    learning_rate = float(os.environ.get("LEARNING_RATE", 1e-4))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", 500))
    lr_scheduler_type = os.environ.get("LR_SCHEDULER_TYPE", "cosine")
    max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 50000))

    optimizer = AdamW(
        unet.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    logger.info(f"Optimizer: AdamW | lr={learning_rate} | weight_decay={weight_decay}")

    logger.info(
        f"LR Scheduler: {lr_scheduler_type} | warmup={warmup_steps} | "
        f"total_steps={max_train_steps}"
    )

    # -------------------------------------------------------------
    # 5. Wrap everything with Accelerate
    # -------------------------------------------------------------

    # Accelerate wraps: UNet, optimizers, LR scheduler
    # It handles: device placement, DDP wrapping, mixed precision scaling
    # We do not wrap VAE or text encoder -- they are frozen and on correct device

    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    logger.info("Accelerate Wrapping Complete")

    # -------------------------------------------------------------
    # 6. Check for checkpoint and resume
    # -------------------------------------------------------------
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_steps = 0
    samples_seen = 0
    epoch = 0
    best_loss = float("inf")

    metadata = load_checkpoint(accelerator, checkpoint_dir)
    if metadata is not None:
        global_steps = metadata["global_steps"]
        samples_seen = metadata["samples_seen"]
        epoch = metadata.get("epoch", 0)
        best_loss = metadata.get("best_loss", float("inf"))
        logger.info(
            f"Resuming from checkpoint | step={global_steps} | "
            f"samples_seen={samples_seen} | epoch={epoch} | "
            f"best_loss={best_loss:.6f}"
        )

    # -------------------------------------------------------------
    # 7. Build DataLoader (after checkpoint load to skip seen)
    # -------------------------------------------------------------
    # The dataloader is built after checkpoint loading because we
    # need samples_seen to skip forward in the streaming dataset
    # We do not wrap the dataloader with Accelerate because
    # it is an IterableDataset -- Accelerate's prepare() would try
    # to use DistributedSampler which does not work with IterableDataset
    # Instead, each GPU process streams independently, With streaming +
    # shuffle, each process gets different ordering, whihc provides
    # natural data parallelism without explicitly sharding

    dataloader = build_dataloader(
        samples_to_skip=samples_seen,
    )

    logger.info(f"DataLoader built | skipped={samples_seen} samples")

    # -------------------------------------------------------------
    # 8. Initialize WandB (only rank 0)
    # -------------------------------------------------------------
    if accelerator.is_main_process:
        wandb_project = os.environ.get("WANDB_PROJECT", "anime-diffusion")
        wandb_mode = os.environ.get("WANDB_MODE", "offline")
        wandb_dir = os.environ.get("WANDB_DIR", "./wandb")
        wandb_run_name = os.environ.get("WANDB_RUN_NAME", "")

        os.makedirs(wandb_dir, exist_ok=True)

        wandb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_steps": warmup_steps,
            "max_train_steps": max_train_steps,
            "batch__size_per_gpu": int(os.environ.get("BATCH_SIZE_PER_GPU", 16)),
            "num_gpus": accelerator.num_processes,
            "effective_batch_size": int(os.environ.get("BATCH_SIZE_PER_GPU", 16))
            * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "image_size": int(os.environ.get("IMAGE_SIZE", 256)),
            "dataset_train_samples": int(
                os.environ.get("DATASET_TRAIN_SAMPLES", 100000)
            ),
            "num_train_timesteps": int(os.environ.get("NUM_TRAIN_TIMESTEPS", 1000)),
            "beta_schedule": os.environ.get("BETA_SCHEDULE", "linear"),
            "grad_clip": float(os.environ.get("GRAD_CLIP", 1.0)),
            "resumed_from_step": global_steps if metadata else 0,
        }

        wandb.init(
            project=wandb_project,
            mode=wandb_mode,
            dir=wandb_dir,
            name=wandb_run_name if wandb_run_name else None,
            config=wandb_config,
            resume="allow",
        )

        logger.info(f"WandB initialized | project={wandb_project} | mode={wandb_mode}")

    # -------------------------------------------------------------
    # 9. Training Loop
    # -------------------------------------------------------------
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY_N_STEPS", 500))
    batch_size = int(os.environ.get("BATCH_SIZE_PER_GPU", 16))
    log_every = 10  # log loss every 10 steps

    # For computing running average loss between checkpoints
    interval_loss = 0.0
    interval_steps = 0

    logger.info(f"Starting training from step={global_steps}")
    logger.info(f"Max steps={max_train_steps} | Checkpoint every={checkpoint_every}")

    train_start_time = time.time()
    step_start_time = time.time()

    unet.train()

    for batch in dataloader:
        if global_steps >= max_train_steps:
            logger.info(f"Reached max steps={max_train_steps}. Stopping training.")
            break

        # Move batch to device (Accelerate does not handle this for unwrapped dataloaders)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        # --- Train one step ---
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

        global_steps += 1
        samples_seen += batch_size * accelerator.num_processes
        interval_loss += loss
        interval_steps += 1

        # --- Logging ---
        if global_steps % log_every == 0:
            avg_loss = interval_loss / interval_steps if interval_steps > 0 else loss
            step_time = (time.time() - step_start_time) / log_every
            current_lr = lr_scheduler.get_last_lr()[0]

            logger.info(
                f"step={global_steps} | loss={loss:.4f} | avg_loss={avg_loss:.4f} | "
                f"lr={current_lr:.2e} | step_time={step_time:.2f}s | "
                f"samples_seen={samples_seen}"
            )

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/avg_loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/step_time": step_time,
                        "train/samples_seen": samples_seen,
                        "train/epoch": epoch,
                    },
                    step=global_steps,
                )

            step_start_time = time.time()

        # --- Checkpoint ---
        if global_steps % checkpoint_every == 0:
            avg_loss_interval = (
                interval_loss / interval_steps if interval_steps > 0 else float("inf")
            )

            best_loss = save_checkpoint(
                accelerator=accelerator,
                checkpoint_dir=checkpoint_dir,
                global_step=global_steps,
                samples_seen=samples_seen,
                epoch=epoch,
                best_loss=best_loss,
                current_loss=avg_loss_interval,
            )

            # Reset interval stats
            interval_loss = 0.0
            interval_steps = 0

    # -------------------------------------------------------------
    # 10. Final checkpoint and cleanup
    # -------------------------------------------------------------

    # Save final checkpoint
    if global_steps % checkpoint_every != 0:
        avg_loss_interval = (
            interval_loss / interval_steps if interval_steps > 0 else float("inf")
        )
        best_loss = save_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=checkpoint_dir,
            global_step=global_steps,
            samples_seen=samples_seen,
            epoch=epoch,
            best_loss=best_loss,
            current_loss=avg_loss_interval,
        )

    total_time = time.time() - train_start_time
    logger.info("-" * 50)
    logger.info("Training Complete")
    logger.info(f"Final step={global_steps}")
    logger.info(f"Total samples seen={samples_seen}")
    logger.info(f"Best Loss={best_loss:.6f}")
    logger.info(f"Total Time={total_time:.2f}s | {total_time / 3600:.2f}h")
    logger.info("-" * 50)

    # Close WandB
    if accelerator.is_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
