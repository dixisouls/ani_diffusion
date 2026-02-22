"""
models/scheduler.py

Noise scheduler for Latent Diffusion Training and Inference.

Two schedulers used:
    - DDPMScheduler (training): 1000 timesteps, linear beta schedule
      Used for add_noise() during training

    - DDIMScheduler (inference): Same beta config but allows fewer steps
      (20-50) for faster sampling. Created seperately at inference time.

The training loop will use:
    noise, timesteps, noisy_latents = sample_noise(scheduler,latents)
    # then: noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)
    # loss = F.mse_loss(noise_pred, noise)

Usage:
    from models.scheduler import build_train_scheduler, build_inference_scheduler
    scheduler = build_train_scheduler()
    noise, timesteps, noisy_latents = sample_noise(scheduler,latents)
"""

import os
import torch
from diffusers import DDPMScheduler, DDIMScheduler

from utils.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------
# Train Scheduler (DDPM)
# -------------------------------------------------------------


def build_train_scheduler() -> DDPMScheduler:
    """
    Build the DDPM noise scheduler used during training.

    Reads from environment:
        - NUM_TRAIN_TIMESTEPS: Number of diffusion steps (default: 1000)
        - BETA_SCHEDULE: Beta schedule type (default: linear)

    The scheduler defines:
        - The variance schedule (how much noise to add at each step)
        - add_noise(): adds noise to clean latents at a given timestep

    Returns:
        DDPMScheduler: The configured scheduler instance for training
    """

    num_train_timesteps = int(os.environ.get("NUM_TRAIN_TIMESTEPS", 1000))
    beta_schedule = os.environ.get("BETA_SCHEDULE", "linear")

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        # clip_sample=False is important!
        # latent values are not bounded to [-1,1] like pixel values,
        # so clipping would corrupt latents during inference
        clip_sample=False,
        # prediction_type="epsilon" -> UNet predicts the noise
        # that was added, not the denoised image directly
        prediction_type="epsilon",
    )

    logger.info(
        f"Training Scheduler ready | type=DDPM | "
        f"timesteps={num_train_timesteps}| beta_schedule={beta_schedule} | "
        f"prediction_type=epsilon"
    )

    return scheduler


def build_inference_scheduler() -> DDIMScheduler:
    """
    Build the DDIM noise scheduler used during inference.

    DDIM allows deterministic, accelerated sampling with far fewer steps
    than the 1000 used in training.

    Uses same beta config as training scheduler to keep noise levels consistent.

    Reads from environment:
        - NUM_TRAIN_TIMESTEPS: Must match training scheduler
        - BETA_SCHEDULE: Must match training scheduler
        - INFERENCE_NUM_STEPS: Number of diffusion steps for sampling (default: 30)

    Returns:
        DDIMScheduler: The configured scheduler instance for inference
    """

    num_train_timesteps = int(os.environ.get("NUM_TRAIN_TIMESTEPS", 1000))
    beta_schedule = os.environ.get("BETA_SCHEDULE", "linear")
    inference_num_steps = int(os.environ.get("INFERENCE_NUM_STEPS", 30))

    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        clip_sample=False,
        prediction_type="epsilon",
    )

    logger.info(
        f"Inference Scheduler ready | type=DDIM | "
        f"timesteps={num_train_timesteps}| beta_schedule={beta_schedule} | "
        f"inference_steps={inference_num_steps} | prediction_type=epsilon"
    )

    return scheduler


def sample_noise(
    scheduler: DDPMScheduler, clean_latents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample noise and timesteps, then create noisy latents.

    Core for forward diffusion process used in training.
        1. Sample gaussian noise matching the latent shape
        2. Sample random timesteps uniformly from [0, T)
        3. Use scheduler.add_noise() to blend clean latents with noise
           according to the variance schedule at each timestep

    Args:
        scheduler: DDPMScheduler instance
        clean_latents: Latent values to add noise to (B,C,H,W)

    Returns:
        noise: Noise added to clean latents (B,C,H,W) -- this is what the UNet will predict
        timesteps: The timesteps sampled for the current batch (B,)
        noisy_latents: Latents with noise added (B,C,H,W
    """

    batch_size = clean_latents.shape[0]
    device = clean_latents.device

    # 1. Sample random gaussian noise
    noise = torch.randn_like(clean_latents)

    # 2. Sample random timesteps, one per sample in the batch
    #    randint is [low,high) so this gives [0,T-1]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )

    # 3. Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    #    scheduler.add_noise handles this math using the precomputed alpha schedule
    noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)

    return noise, timesteps, noisy_latents
