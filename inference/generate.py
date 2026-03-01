"""
inference/generate.py

Manual inference pipeline for ani_diffusion.

This is the explicit, step-by-step pipeline where every operation
is visible. Use this to understand exactly what happens during
image generation, or when you need fine-grained control over the
denoising process.

Pipeline steps:
    1. Encode text prompt with CLIP (and empty prompt for CFG)
    2. Generate initial random noise in latent space
    3. Iteratively denoise using DDIM scheduler + UNet
    4. Apply classifier-free guidance (CFG) at each step
    5. Decode final latent with VAE to get pixel-space image
    6. Post-process and save as PNG

Supports:
    - Classifier-free guidance (CFG) for stronger prompt adherence
    - EMA weight loading from checkpoint
    - Batch generation of multiple images
    - Reproducible generation with seed control

Usage (CLI):
    python inference/generate.py \\
        --prompt "a girl with blue hair in a garden" \\
        --checkpoint_dir ./checkpoints \\
        --output_dir ./outputs \\
        --num_images 4 \\
        --seed 42
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------
# Pipeline Loading
# ------------------------------------------------------------


def load_pipeline(
    checkpoint_dir: str,
    device: str = "cuda",
    use_ema: bool = True,
) -> dict:
    """
    Load all model components needed for inference.

    Loads VAE, text encoder, tokenizer from pretrained sources,
    then loads the trained UNet from checkpoint. If EMA is available
    and use_ema=True, applies EMA shadow weights to the UNet.

    Args:
        checkpoint_dir: Path to the root checkpoint directory
                        (contains metadata.json, step_* dirs, best/ dir)
        device: Device to load models onto ("cuda" or "cpu")
        use_ema: Whether to apply EMA weights (recommended for quality)

    Returns:
        dict with keys: vae, text_encoder, tokenizer, unet, scheduler, device, dtype
    """
    from models.loader import load_models
    from models.scheduler import build_inference_scheduler
    from training.checkpoint import find_latest_checkpoint, load_ema_state
    from training.ema import EMAModel

    # --- Load pretrained components ---
    logger.info("Loading model components for inference...")
    vae, text_encoder, tokenizer, unet = load_models()

    # --- Find and load checkpoint ---
    # Prefer 'best' checkpoint, fall back to latest
    best_path = os.path.join(checkpoint_dir, "best")

    if os.path.exists(best_path):
        checkpoint_path = best_path
        logger.info(f"Using best checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            logger.error(
                "No checkpoint found. Cannot run inference without trained weights."
            )
            sys.exit(1)
        logger.info(f"Using latest checkpoint: {checkpoint_path}")

    # --- Load UNet weights from checkpoint ---
    unet_state_path = find_unet_state(checkpoint_path)
    if unet_state_path is not None:
        logger.info(f"Loading UNet state from: {unet_state_path}")
        if unet_state_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(unet_state_path, device="cpu")
        else:
            state_dict = torch.load(unet_state_path, map_location="cpu", weights_only=False)
        unet.load_state_dict(state_dict)
        logger.info("UNet checkpoint weights loaded")
    else:
        logger.warning(
            "Could not find UNet state dict in checkpoint. "
            "Using initialization weights -- output quality will be poor."
        )

    # --- Apply EMA weights if available ---
    if use_ema:
        ema_model = EMAModel(unet, decay=0.9999)
        ema_loaded = load_ema_state(checkpoint_path, ema_model)
        if ema_loaded:
            ema_model.apply_shadow(unet)
            logger.info("EMA shadow weights applied to UNet for inference")
        else:
            logger.warning("No EMA state found -- using raw training weights")

    # --- Move to device and set eval mode ---
    weight_dtype = torch.float32 if device == "cpu" else torch.float16

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    vae.eval()
    text_encoder.eval()
    unet.eval()

    # --- Build inference scheduler ---
    scheduler = build_inference_scheduler()

    logger.info(f"Inference pipeline ready | device={device} | dtype={weight_dtype}")

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "unet": unet,
        "scheduler": scheduler,
        "device": device,
        "dtype": weight_dtype,
    }


# ------------------------------------------------------------
# UNet State Finder
# ------------------------------------------------------------


def find_unet_state(checkpoint_path: str) -> str | None:
    """
    Locate the UNet state dict file within an Accelerate checkpoint.

    Accelerate's save_state() saves model weights in a subdirectory.
    The exact structure depends on the accelerate version:
        - Newer: checkpoint_path/model_0/model.safetensors
        - Older: checkpoint_path/pytorch_model.bin

    Args:
        checkpoint_path: Path to the specific checkpoint directory

    Returns:
        Path to the UNet state file, or None if not found
    """
    candidates = [
        os.path.join(checkpoint_path, "model_0", "model.safetensors"),
        os.path.join(checkpoint_path, "model_0", "pytorch_model.bin"),
        os.path.join(checkpoint_path, "pytorch_model.bin"),
        os.path.join(checkpoint_path, "model.safetensors"),
        os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.safetensors"),
        os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.bin"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Log checkpoint contents for debugging
    logger.warning(f"No UNet state found. Checkpoint contents: {os.listdir(checkpoint_path)}")
    for item in os.listdir(checkpoint_path):
        sub = os.path.join(checkpoint_path, item)
        if os.path.isdir(sub):
            logger.warning(f"  {item}/: {os.listdir(sub)}")

    return None


# ------------------------------------------------------------
# Text Encoding (with CFG support)
# ------------------------------------------------------------


@torch.no_grad()
def encode_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    device: str,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Encode text prompt into embeddings for the UNet.

    For classifier-free guidance (CFG), encodes both the real prompt
    and an empty string. The two are concatenated along the batch
    dimension so the UNet can process both in a single forward pass.

    CFG formula (applied per denoising step):
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    Args:
        prompt: The text description of the desired image
        tokenizer: CLIP tokenizer
        text_encoder: Frozen CLIP text encoder
        device: Target device
        dtype: Target dtype (float16 for GPU inference)
        do_classifier_free_guidance: Whether to also encode empty prompt

    Returns:
        prompt_embeds: (2, 77, 768) if CFG enabled, (1, 77, 768) otherwise
    """
    max_length = int(os.environ.get("TOKENIZER_MAX_LENGTH", 77))

    # --- Encode the real prompt ---
    text_inputs = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    prompt_embeds = text_encoder(
        input_ids=text_inputs.input_ids.to(device),
        attention_mask=text_inputs.attention_mask.to(device),
    ).last_hidden_state

    prompt_embeds = prompt_embeds.to(dtype=dtype)

    # --- Encode empty prompt for CFG ---
    if do_classifier_free_guidance:
        uncond_input = tokenizer(
            "",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        uncond_embeds = text_encoder(
            input_ids=uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
        ).last_hidden_state

        uncond_embeds = uncond_embeds.to(dtype=dtype)

        # Concatenate: [unconditional, conditional] along batch dim
        prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)

    return prompt_embeds


# ------------------------------------------------------------
# Core Generation Function
# ------------------------------------------------------------


@torch.no_grad()
def generate_images(
    pipeline: dict,
    prompt: str,
    num_images: int = 1,
    num_inference_steps: int = None,
    guidance_scale: float = 7.5,
    seed: int = None,
) -> list[Image.Image]:
    """
    Generate images from a text prompt using the trained model.

    For each denoising step:
        1. Duplicate the noisy latent (for CFG: one uncond, one cond)
        2. UNet predicts noise for both
        3. Apply CFG: blend uncond and cond predictions
        4. Scheduler computes the denoised latent for the next step
    After all steps, decode the final latent with VAE.

    Args:
        pipeline: dict from load_pipeline()
        prompt: Text description of the desired image
        num_images: Number of images to generate (batch size)
        num_inference_steps: Denoising steps (default from env: 30)
        guidance_scale: CFG strength. 1.0 = no guidance, 7.5 = standard
        seed: Random seed for reproducibility (None = random)

    Returns:
        List of PIL Images (256x256 RGB)
    """
    vae = pipeline["vae"]
    text_encoder = pipeline["text_encoder"]
    tokenizer = pipeline["tokenizer"]
    unet = pipeline["unet"]
    scheduler = pipeline["scheduler"]
    device = pipeline["device"]
    dtype = pipeline["dtype"]

    if num_inference_steps is None:
        num_inference_steps = int(os.environ.get("INFERENCE_NUM_STEPS", 30))

    do_cfg = guidance_scale > 1.0

    # --- Seed for reproducibility ---
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        logger.info(f"Using seed: {seed}")

    # --- Encode prompt ---
    prompt_embeds = encode_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
        do_classifier_free_guidance=do_cfg,
    )

    # --- Expand prompt embeds to match full CFG batch size (2 * num_images) ---
    if do_cfg:
        uncond_embeds, cond_embeds = prompt_embeds.chunk(2)
        prompt_embeds = torch.cat([
            uncond_embeds.repeat(num_images, 1, 1),
            cond_embeds.repeat(num_images, 1, 1),
        ])
    else:
        prompt_embeds = prompt_embeds.repeat(num_images, 1, 1)

    # --- Generate initial noise in latent space ---
    latent_channels = int(os.environ.get("LATENT_CHANNELS", 4))
    latent_size = int(os.environ.get("LATENT_SIZE", 32))

    latents = torch.randn(
        (num_images, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # --- Set up scheduler timesteps ---
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Scale initial noise by scheduler's init_noise_sigma
    latents = latents * scheduler.init_noise_sigma

    # --- Denoising loop ---
    logger.info(
        f"Generating {num_images} image(s) | steps={num_inference_steps} | "
        f"guidance_scale={guidance_scale}"
    )

    for i, t in enumerate(tqdm(timesteps, desc="Denoising", disable=False)):
        # Expand latents for CFG: [uncond_latent, cond_latent]
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

        # Some schedulers require scaling the input at each step
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # --- UNet predicts noise ---
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample

        # --- Apply classifier-free guidance ---
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        # --- Scheduler step: compute x_{t-1} from x_t ---
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # --- Decode latents to pixel space ---
    images = decode_latents(latents, vae)

    logger.info(f"Generation complete | {len(images)} image(s)")
    return images


# ------------------------------------------------------------
# Latent Decoding
# ------------------------------------------------------------


@torch.no_grad()
def decode_latents(latents: torch.Tensor, vae) -> list[Image.Image]:
    """
    Decode latent tensors to PIL Images using the VAE decoder.

    The VAE encoder multiplied by scaling_factor (0.18215) during training,
    so we divide by it here to undo that scaling before decoding.

    Args:
        latents: (B, 4, 32, 32) latent tensor
        vae: Frozen AutoencoderKL

    Returns:
        List of PIL Images (256x256 RGB)
    """
    latents = latents / vae.config.scaling_factor

    decoded = vae.decode(latents.to(dtype=vae.dtype)).sample

    # [-1, 1] -> [0, 255] uint8
    images = (decoded / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype(np.uint8)

    pil_images = [Image.fromarray(img) for img in images]
    return pil_images


# ------------------------------------------------------------
# Save Helper
# ------------------------------------------------------------


def save_images(
    images: list[Image.Image],
    output_dir: str,
    prompt: str,
    seed: int = None,
):
    """
    Save generated images to disk with descriptive filenames.

    Args:
        images: List of PIL Images
        output_dir: Directory to save images
        prompt: The text prompt (saved alongside for reference)
        seed: The seed used (included in filename)
    """
    os.makedirs(output_dir, exist_ok=True)

    seed_str = f"seed{seed}" if seed is not None else "seedrandom"

    for i, img in enumerate(images):
        filename = f"{seed_str}_{i:02d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        logger.info(f"Saved: {filepath}")

    prompt_file = os.path.join(output_dir, f"{seed_str}_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Images: {len(images)}\n")


# ------------------------------------------------------------
# Env File Loader (standalone CLI use outside SLURM)
# ------------------------------------------------------------


def _load_env_file(path: str):
    """Load a shell-style env file (export KEY=VALUE lines)."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                os.environ[key.strip()] = value


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate anime images (manual pipeline)"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./outputs"))
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    args = parser.parse_args()

    # Load env if running outside SLURM
    if "HF_DATASET_ID" not in os.environ:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.env",
        )
        if os.path.exists(config_path):
            _load_env_file(config_path)

    pipeline = load_pipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_ema=not args.no_ema,
    )

    images = generate_images(
        pipeline=pipeline,
        prompt=args.prompt,
        num_images=args.num_images,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    save_images(images=images, output_dir=args.output_dir, prompt=args.prompt, seed=args.seed)
    logger.info("Done.")


if __name__ == "__main__":
    main()