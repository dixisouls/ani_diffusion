"""
models/loader.py

Model loading for the diffusion model.
Loads and configures all three model components:
    - VAE : frozen, used to encode images to latents and decode back
    - Text Encoder : frozen, used to encode captions to text embeddings
    - UNet : trainable, the core denoising network

Design Decisions:
    - All models loaded to CPU first, Accelerate handles device placement later
    - VAE and text encode are fully frozen (no gradient, eval mode)
    - UNet is fully trainable (all gradients enabled, train mode)
    - Mixed precision is handled by Accelerate, load in float32

Usage:
    from models.loader import load_models
    vae, text_encoder, tokenizer, unet = load_models()
"""

import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

from utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Freeze Helper
# -----------------------------------------------------------------------------


def freeze_model(model):
    """
    Freeze all parameters of a model.
    - Disables gradient computation on all parameters.
    - Sets model to eval mode (disable dropout, uses running batch norm stats)
    """

    model.eval()
    for param in model.parameters():
        param.requires_grad = False


# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------


def load_models():
    """
    Load and configure all model components for training.

    Returns:
        - vae : AutoencoderKL, frozen
        - text_encoder : CLIPTextModel, frozen
        - tokenizer : CLIPTokenizer
        - unet : UNet2DConditionModel, trainable

    All models are on CPU. Accelerate moves them to correct device.
    """

    vae_model_id = os.environ["VAE_MODEL_ID"]
    text_encoder_model_id = os.environ["TEXT_ENCODER_MODEL_ID"]
    tokenizer_model_id = os.environ["TOKENIZER_MODEL_ID"]
    unet_model_id = os.environ["UNET_MODEL_ID"]

    # -------------------------------------------------------------
    # VAE
    # -------------------------------------------------------------

    logger.info(f"Loading VAE from {vae_model_id}")
    vae = AutoencoderKL.from_pretrained(vae_model_id)
    freeze_model(vae)
    logger.info(
        f"VAE loaded | frozen | "
        f"params: {sum(p.numel() for p in vae.parameters()):,}"
    )

    # -------------------------------------------------------------
    # Text Encoder and Tokenizer
    # -------------------------------------------------------------
    logger.info(f"Loading Text Encoder from {text_encoder_model_id}")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_model_id)
    freeze_model(text_encoder)
    logger.info(
        f"Text Encoder loaded | frozen | "
        f"params: {sum(p.numel() for p in text_encoder.parameters()):,}"
    )

    logger.info(f"Loading Tokenizer from {tokenizer_model_id}")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model_id)
    logger.info("Tokenizer loaded")

    # -------------------------------------------------------------
    # UNet
    # -------------------------------------------------------------

    logger.info(f"Loading UNet from {unet_model_id}")
    unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet")
    unet.train()
    logger.info(
        f"UNet loaded | trainable | "
        f"params: {sum(p.numel() for p in unet.parameters()):,} | "
        f"trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}"
    )

    return vae, text_encoder, tokenizer, unet


# -----------------------------------------------------------------------------
# Parameter Count Helper (useful for sanity checks)
# -----------------------------------------------------------------------------


def count_parameters(model):
    """
    Return total and trainable parameter counts for a model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
