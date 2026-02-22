"""
tests/test_module_2.py

Verifies that all model components are loaded correctly.

Tests:
1. All models load without errors
2. VAE and text encoder are fully frozen
3. UNet is fully trainable
4. Parameter counts are as expected ranges
5. VAE encodes a dummy image to correct latent shape
6. Text encoder encodes a dummy caption to correct embedding shape

Run via:
    sbatch test_module_2.slurm
"""

import os
import sys
import torch
from utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Starting test_module_2")

# -------------------------------------------------------------
# Load Models
# -------------------------------------------------------------

try:
    from models.loader import load_models, count_parameters
    vae, text_encoder, tokenizer, unet = load_models()
    logger.info("[PASSED] Models loaded successfully")
except Exception as e:
    logger.error(f"[FAILED] Failed to load models: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# VAE and Text Encoder are frozen
# -------------------------------------------------------------

try:
    vae_params = count_parameters(vae)
    text_encoder_params = count_parameters(text_encoder)
    
    assert vae_params['trainable'] == 0, \
        f"VAE has {vae_params['trainable']:,} trainable parameters, expected 0"
    assert text_encoder_params['trainable'] == 0, \
        f"Text Encoder has {text_encoder_params['trainable']:,} trainable parameters, expected 0"
    
    # Also verify eval mode
    assert not vae.training, "VAE should be in eval mode"
    assert not text_encoder.training, "Text Encoder should be in eval mode"
    
    
    logger.info(f"VAE : total={vae_params['total']:,} | trainable={vae_params['trainable']:,}")
    logger.info(f"Text Encoder : total={text_encoder_params['total']:,} | trainable={text_encoder_params['trainable']:,}")
    logger.info("[PASSED] VAE and Text Encoder are frozen and in eval mode")

except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)


# -------------------------------------------------------------
# UNet is trainable
# -------------------------------------------------------------

try:
    unet_params = count_parameters(unet)
    assert unet_params['trainable'] > 0, "UNet has no trainable parameters"
    assert unet_params['trainable'] == unet_params['total'], \
        f"UNet has frozen parameters: total={unet_params['total']:,} | trainable={unet_params['trainable']:,}"
    assert unet.training, "UNet should be in train mode"
    
    logger.info(f"UNet : total={unet_params['total']:,} | trainable={unet_params['trainable']:,}")
    logger.info("[PASSED] UNet is trainable and in train mode")

except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)


# -------------------------------------------------------------
# VAE encodes a dummy image to correct latent shape
# -------------------------------------------------------------

try:
    image_size = int(os.environ["IMAGE_SIZE"]) # 256
    latent_size = int(os.environ["LATENT_SIZE"]) # 32
    latent_channels = int(os.environ["LATENT_CHANNELS"]) # 4

    # Dummy image batch: (1, 3, 256, 256) normalized to [-1, 1]
    dummy_image = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        latent_dist = vae.encode(dummy_image)
        # vae returns a distribution, sample from it
        latents = latent_dist.latent_dist.sample()
        # SD vae uses scaling factor of 0.18215
        latents = latents * vae.config.scaling_factor
    
    expected_shape = torch.Size([1, latent_channels, latent_size, latent_size])
    assert latents.shape == expected_shape, \
        f"Latent shape mismatch: {latents.shape} != {expected_shape}"
    
    logger.info(f"Dummy image shape: {list(dummy_image.shape)}")
    logger.info(f"Latent shape: {list(latents.shape)}")
    logger.info(f"Latent min/max : {latents.min():.4f}/{latents.max():.4f}")
    logger.info("[PASSED] VAE encodes to correct latent shape")

except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)


# -------------------------------------------------------------
# Text encoder produces correct embedding shape
# -------------------------------------------------------------

try:
    max_length = int(os.environ["TOKENIZER_MAX_LENGTH"]) # 77
    dummy_caption = "a young girl with blue hair standing in her garden"
    tokens = tokenizer(
        dummy_caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        embeddings = text_encoder(
            tokens.input_ids,
            attention_mask=tokens.attention_mask
        )
        
        # CLIPTextModel returns a BaseModelOutputWithPooling
        # last hidden state is what UNet cross-attention expects
        hidden_states = embeddings.last_hidden_state
    
    assert hidden_states.shape[0] == 1, f"Batch dim mismatch: {hidden_states.shape[0]}"
    assert hidden_states.shape[1] == max_length, f"Sequence length mismatch: {hidden_states.shape[1]}"
    assert hidden_states.shape[2] > 0, f"Hidden dim is 0"

    logger.info(f"Dummy caption: {dummy_caption}")
    logger.info(f"Embeddings: shape={list(hidden_states.shape)}")
    logger.info(f"Hidden dim: {hidden_states.shape[2]} (CLIP ViT-L/14 = 768)")
    logger.info("[PASSED] Text encoder produces correct embedding shape")

except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)


logger.info("All tests passed successfully")
sys.exit(0)