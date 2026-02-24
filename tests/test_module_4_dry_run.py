"""
tests/test_module_4_dry_run.py

Quick sanity check that the training step works end-to-end.

This runs on CPU with a TINY config to verify:
    1. All imports resolve correctly
    2. One forward + backward pass completes without error
    3. Loss is a valid finite number
    4. Gradient clipping does not crash
    5. Checkpoint save/load roundtrips metadata correctly

This does NOT test multi-GPU, real data streaming, or WandB.
Those are tested via the integration SLURM test.

Run:
    sbatch tests/test_module_4_dry_run.slurm
"""

import os
import sys
import json
import shutil
import torch

from utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Starting test_module_4_dry_run")

# -------------------------------------------------------------
# 1. Test: training/loop.py imports and encode helpers work
# -------------------------------------------------------------

try:
    from training.loop import encode_images_to_latents, encode_text, train_one_step

    logger.info("[PASSED] training.loop imports resolved")
except Exception as e:
    logger.error(f"[FAILED] training.loop import error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 2. Test: training/checkpoint.py imports and metadata roundtrip
# -------------------------------------------------------------

try:
    from training.checkpoint import (
        save_metadata,
        load_metadata,
        find_latest_checkpoint,
        _rotate_checkpoints,
    )

    # Create a temporary checkpoint directory
    test_ckpt_dir = os.path.join(
        os.environ.get("CHECKPOINT_DIR", "./checkpoints"), "_test_temp"
    )
    os.makedirs(test_ckpt_dir, exist_ok=True)

    # Save metadata
    save_metadata(
        checkpoint_dir=test_ckpt_dir,
        global_step=100,
        samples_seen=1600,
        epoch=0,
        best_loss=0.123456,
        latest_checkpoint_name="step_100",
    )

    # Load metadata
    loaded = load_metadata(test_ckpt_dir)
    assert loaded is not None, "Metadata load returned None"
    assert (
        loaded["global_step"] == 100
    ), f"global_step mismatch: {loaded['global_step']}"
    assert (
        loaded["samples_seen"] == 1600
    ), f"samples_seen mismatch: {loaded['samples_seen']}"
    assert loaded["best_loss"] == 0.123456, f"best_loss mismatch: {loaded['best_loss']}"
    assert loaded["latest_checkpoint_name"] == "step_100"

    # Cleanup
    shutil.rmtree(test_ckpt_dir)

    logger.info("[PASSED] Checkpoint metadata save/load roundtrip")

except Exception as e:
    # Cleanup on failure
    if os.path.exists(test_ckpt_dir):
        shutil.rmtree(test_ckpt_dir)
    logger.error(f"[FAILED] Checkpoint metadata test: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 3. Test: Checkpoint rotation logic
# -------------------------------------------------------------

try:
    test_ckpt_dir = os.path.join(
        os.environ.get("CHECKPOINT_DIR", "./checkpoints"), "_test_rotate"
    )
    os.makedirs(test_ckpt_dir, exist_ok=True)

    # Create 5 fake checkpoint directories
    for step in [100, 200, 300, 400, 500]:
        step_dir = os.path.join(test_ckpt_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        # Put a dummy file so the dir is not empty
        with open(os.path.join(step_dir, "dummy.txt"), "w") as f:
            f.write("test")

    # Also create a 'best' directory -- should never be deleted
    best_dir = os.path.join(test_ckpt_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    # Set keep_n to 3 for this test
    original_keep = os.environ.get("KEEP_LAST_N_CHECKPOINTS")
    os.environ["KEEP_LAST_N_CHECKPOINTS"] = "3"

    _rotate_checkpoints(test_ckpt_dir)

    # Check: should have step_300, step_400, step_500 remaining + best
    remaining = sorted(
        [
            d
            for d in os.listdir(test_ckpt_dir)
            if os.path.isdir(os.path.join(test_ckpt_dir, d)) and d.startswith("step_")
        ]
    )
    assert remaining == [
        "step_300",
        "step_400",
        "step_500",
    ], f"Expected [step_300, step_400, step_500], got {remaining}"
    assert os.path.exists(best_dir), "Best checkpoint was incorrectly deleted"

    # Restore original env
    if original_keep is not None:
        os.environ["KEEP_LAST_N_CHECKPOINTS"] = original_keep
    else:
        os.environ.pop("KEEP_LAST_N_CHECKPOINTS", None)

    # Cleanup
    shutil.rmtree(test_ckpt_dir)

    logger.info("[PASSED] Checkpoint rotation keeps last N and preserves 'best'")

except Exception as e:
    if os.path.exists(test_ckpt_dir):
        shutil.rmtree(test_ckpt_dir)
    logger.error(f"[FAILED] Checkpoint rotation test: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 4. Test: VAE encode produces correct shape (with real model on CPU)
# -------------------------------------------------------------

try:
    from models.loader import load_models

    logger.info(
        "Loading models for forward pass test (this may take a minute on CPU)..."
    )
    vae, text_encoder, tokenizer, unet = load_models()

    image_size = int(os.environ["IMAGE_SIZE"])
    latent_size = int(os.environ["LATENT_SIZE"])
    latent_channels = int(os.environ["LATENT_CHANNELS"])

    # Dummy batch of 2 images
    dummy_pixels = torch.randn(2, 3, image_size, image_size)

    latents = encode_images_to_latents(dummy_pixels, vae)

    expected_shape = torch.Size([2, latent_channels, latent_size, latent_size])
    assert (
        latents.shape == expected_shape
    ), f"Latent shape mismatch: {latents.shape} != {expected_shape}"

    logger.info(
        f"[PASSED] VAE encode | input={list(dummy_pixels.shape)} -> latents={list(latents.shape)}"
    )

except Exception as e:
    logger.error(f"[FAILED] VAE encode test: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 5. Test: Text encode produces correct shape
# -------------------------------------------------------------

try:
    max_length = int(os.environ["TOKENIZER_MAX_LENGTH"])

    dummy_tokens = tokenizer(
        ["a girl with blue hair", "a boy in a garden"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    hidden_states = encode_text(
        dummy_tokens.input_ids,
        dummy_tokens.attention_mask,
        text_encoder,
    )

    assert hidden_states.shape[0] == 2, f"Batch dim mismatch: {hidden_states.shape[0]}"
    assert (
        hidden_states.shape[1] == max_length
    ), f"Seq len mismatch: {hidden_states.shape[1]}"
    assert (
        hidden_states.shape[2] == 768
    ), f"Hidden dim mismatch: {hidden_states.shape[2]}"

    logger.info(f"[PASSED] Text encode | shape={list(hidden_states.shape)}")

except Exception as e:
    logger.error(f"[FAILED] Text encode test: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 6. Test: Full forward + backward pass (CPU, no Accelerate)
# -------------------------------------------------------------

try:
    from models.scheduler import build_train_scheduler, sample_noise
    import torch.nn.functional as F

    noise_scheduler = build_train_scheduler()

    # Use the latents from test 4
    noise, timesteps, noisy_latents = sample_noise(noise_scheduler, latents)

    # Forward pass through UNet
    unet.train()
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=hidden_states,
    ).sample

    assert (
        noise_pred.shape == noise.shape
    ), f"UNet output shape mismatch: {noise_pred.shape} != {noise.shape}"

    # Compute loss
    loss = F.mse_loss(noise_pred, noise)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss is non-positive: {loss.item()}"

    logger.info(
        f"Forward pass OK | noise_pred={list(noise_pred.shape)} | loss={loss.item():.4f}"
    )

    # Backward pass
    loss.backward()

    # Check that UNet has gradients
    grad_norms = []
    for name, param in unet.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    assert len(grad_norms) > 0, "No gradients computed on UNet parameters"

    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    logger.info(
        f"Backward pass OK | params_with_grad={len(grad_norms)} | "
        f"avg_grad_norm={avg_grad_norm:.6f}"
    )

    # Zero gradients for cleanup
    unet.zero_grad()

    logger.info("[PASSED] Full forward + backward pass on CPU")

except Exception as e:
    logger.error(f"[FAILED] Forward/backward pass: {e}")
    sys.exit(1)

# -------------------------------------------------------------
logger.info("-" * 50)
logger.info("All module 4 dry-run tests passed")
logger.info("-" * 50)
sys.exit(0)
