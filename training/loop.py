"""
training/loop.py

Single training step logic for Latent Diffusion Model.

This module contains the core forward-backward pass:
    1. Encode pixel values to latent using frozen VAE
    2. Encode captions to embeddings using frozen CLIP text encoder
    3. Sample noise and timesteps via noise scheduler
    4. UNet predicts noise from noisy latents + timesteps + text embeddings
    5. Compute MSE loss between predicted and true noise
    6. Backward pass and optimizer step

The training orchestrator (train.py) calls train_one_step() for each batch.
This seperation keeps train.py focused on orchestration and mkes the
step logic independent testable.

Usage:
    from training.loop import train_one_step
    loss = train_one_step(
        batch, unet, vae, text_encoder,
        noise_scheduler, optimizer, lr_scheduler, accelerator
    )
"""

import torch
import torch.nn.functional as F

from models.scheduler import sample_noise
from utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------
# VAE Encoding (frozen, no gradients)
# ------------------------------------------------------------
@torch.no_grad()
def encode_images_to_latents(pixel_values: torch.Tensor, vae) -> torch.Tensor:
    """
    Encode pixel-space images to VAE latent space.

    The VAE encoder compresses 256x256x3 images to 32x32x4 latents.
    We multiply by the scaling factor (0.18215 for SD's VAE) to
    normalize the latent distribution. This ensures the latent space
    has rougly unit variance, which the noise scheduler expects.

    Args:
        pixel_values: (B, 3, 256, 256) tensor normalized to [-1, 1]
        vae: Frozen AutoencoderKL instance

    Returns:
        latents: (B, 4, 32, 32) tensor -- scaled latent representation
    """

    latent_dist = vae.encode(pixel_values).latent_dist
    latents = latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


# ------------------------------------------------------------
# Text Encoding (frozen, no gradients)
# ------------------------------------------------------------
@torch.no_grad()
def encode_text(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, text_encoder
) -> torch.Tensor:
    """
    Encode tokenized captions to CLIP text embeddings.

    Returns the last hidden state from CLIP, which is the sequence of token embeddings
    the UNet cross-attention layers attend to.

    Args:
        input_ids: (B, 77) token IDs from CLIP tokenizer
        attention_mask: (B, 77) attention mask (1 = real token, 0 = padding)
        text_encoder: Frozen CLIPTextModel instance

    Returns:
        encoder_hidden_states: (B, 77, 768) tensor -- text embeddings for cross-attention
    """

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return outputs.last_hidden_state


# ------------------------------------------------------------
# Single Training Step
# ------------------------------------------------------------
def train_one_step(
    batch: dict,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    accelerator,
) -> float:
    """
    Execute one single training step: forward pass, loss, backward, optimize.

    This the core training logic called once per batch. The sequence is :
        1. Encode images to latents (VAE, frozen, no grad)
        2. Encode captions to embeddings (CLIP, frozen, no grad)
        3. Sample noise and timesteps
        4. Create noisy latents
        5. UNet predicts noise
        6. MSE loss between predicted and true noise
        7. Backward pass (Accelerate handles gradient sync across devices)
        8. Gradient clipping
        9. Optimizer step
        10. LR scheduler step

    Args:
        batch: dict with 'pixel_values', 'input_ids', 'attention_mask'
        unet: accelerate-wrapped UNet2DConditionModel (trainable)
        vae: frozen AutoencoderKL instance (on correct device via Accelerate)
        text_encoder: frozen CLIPTextModel instance (on correct device via Accelerate)
        noise_scheduler: DDPMScheduler instance
        optimizer: accelerate-wrapped optimizer
        lr_scheduler: learning rate scheduler
        accelerator: accelerate instance (for backward pass and gradient clipping)

    Returns:
        loss_value: float -- the MSE loss for this step (detached, for logging)
    """

    import os

    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))

    # 1. Encode images to latents (no grad, VAE frozen)
    pixel_values = batch["pixel_values"]
    latents = encode_images_to_latents(pixel_values, vae)

    # 2. Encode captions to embeddings (no grad, CLIP frozen)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    encoder_hidden_states = encode_text(input_ids, attention_mask, text_encoder)

    # 3 and 4. Sample noise and timesteps, create noisy latents
    noise, timesteps, noisy_latents = sample_noise(noise_scheduler, latents)

    # 5. UNet predicts noise
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
    ).sample

    # 6. MSE loss between predicted and true noise
    loss = F.mse_loss(noise_pred, noise)

    # 7. Backward pass (Accelerate handles gradient sync)
    accelerator.backward(loss)

    # 8. Gradient clipping (applied to UNets parameters only)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet.parameters(), grad_clip)

    # 9. Optimizer step
    optimizer.step()

    # 10. LR scheduler step
    lr_scheduler.step()

    # Clear gradients
    optimizer.zero_grad()

    # Return detached loss
    return loss.detach().item()
