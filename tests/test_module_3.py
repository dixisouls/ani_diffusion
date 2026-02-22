"""
tests/test_module_3.py

Verifies the noise scheduler works correctly.

Tests:
1. Training scheduler instantiation
2. Inference scheduler instantiation
3. sample_noise() produces correct shapes and devices
4. Noise latents differ from clean latents (noise is actually added)
5. Noise magnitude increases with each timestep (variance schedule is monotonic)
6. add_noise() is deterministic for fixed noise and timesteps

Run:
    sbatch tests/test_module_3.slurm
"""

import os
import sys
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------
# Training Scheduler Instantiation
# -------------------------------------------------------------

try:
    from models.scheduler import build_train_scheduler
    scheduler = build_train_scheduler()
    
    expected_timesteps = int(os.environ["NUM_TRAIN_TIMESTEPS"])
    expected_beta_schedule = os.environ["BETA_SCHEDULE"]

    assert scheduler.config.num_train_timesteps == expected_timesteps, \
        f"Timesteps mismatch: {scheduler.config.num_train_timesteps} != {expected_timesteps}"
    assert scheduler.config.beta_schedule == expected_beta_schedule, \
        f"Beta schedule mismatch: {scheduler.config.beta_schedule} != {expected_beta_schedule}"
    assert scheduler.config.prediction_type == "epsilon", \
        f"Prediction type mismatch: {scheduler.config.prediction_type} != epsilon"
    assert scheduler.config.clip_sample == False, \
        f"Clip sample mismatch: {scheduler.config.clip_sample} != False"
    
    logger.info(
        f"[PASSED] Training Scheduler Instantiation | "
        f"timesteps={scheduler.config.num_train_timesteps} | "
        f"beta_schedule={scheduler.config.beta_schedule}"
    )

except AssertionError as e:
    logger.error(f"[FAILED] Training Scheduler Instantiation | Error: {e}")
    sys.exit(1)

except Exception as e:
    logger.error(f"[FAILED] Training Scheduler Instantiation | Unexpected Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# Inference Scheduler Instantiation
# -------------------------------------------------------------

try:
    from models.scheduler import build_inference_scheduler
    inf_scheduler = build_inference_scheduler()
    
    assert inf_scheduler.config.num_train_timesteps == expected_timesteps, \
        f"Inference timesteps mismatch: {inf_scheduler.config.num_train_timesteps} != {expected_timesteps}"
    assert inf_scheduler.config.beta_schedule == expected_beta_schedule, \
        f"Inference beta schedule mismatch: {inf_scheduler.config.beta_schedule} != {expected_beta_schedule}"
    assert inf_scheduler.config.clip_sample == False, \
        f"Inference clip sample mismatch: {inf_scheduler.config.clip_sample} != False"
    
    logger.info("[PASSED] Inference Scheduler Instantiation")

except AssertionError as e:
    logger.error(f"[FAILED] Inference Scheduler Instantiation | Error: {e}")
    sys.exit(1)

except Exception as e:
    logger.error(f"[FAILED] Inference Scheduler Instantiation | Unexpected Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# sample_noise() produces correct shapes and devices
# -------------------------------------------------------------
try:
    from models.scheduler import sample_noise
    
    # create a fake latent batch: (B=4,C=4,H=32,W=32)
    latent_channels = int(os.environ["LATENT_CHANNELS"])
    latent_size = int(os.environ["LATENT_SIZE"])
    batch_size = 4

    clean_latents = torch.randn(batch_size, latent_channels, latent_size, latent_size)

    noise, timesteps, noisy_latents = sample_noise(scheduler, clean_latents)

    assert noise.shape == clean_latents.shape, \
        f"Noise shape mismatch: {noise.shape} != {clean_latents.shape}"
    assert noisy_latents.shape == clean_latents.shape, \
        f"Noisy latents shape mismatch: {noisy_latents.shape} != {clean_latents.shape}"
    assert timesteps.shape == (batch_size,), \
        f"Timesteps shape mismatch: {timesteps.shape} != ({batch_size},)"
    

    # Dtype checks
    assert timesteps.dtype == torch.long, \
        f"Timesteps dtype mismatch: {timesteps.dtype} != torch.long"
    
    # Range check: timesteps should be in [0, T)
    assert timesteps.min() >= 0, f"Timesteps below 0: {timesteps.min()}"
    assert timesteps.max() < expected_timesteps, f"Timesteps above T-1: {timesteps.max()} | expected max: {expected_timesteps-1}"

    logger.info(
        f"[PASSED] sample_noise shapes | noise={list(noise.shape)} | "
        f"timesteps={list(timesteps.shape)} | noisy_latents={list(noisy_latents.shape)}"
    )

except AssertionError as e:
    logger.error(f"[FAILED] sample_noise shapes | Error: {e}")
    sys.exit(1)

except Exception as e:
    logger.error(f"[FAILED] sample_noise shapes | Unexpected Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# Noise latents differ from clean latents (noise is actually added)
# -------------------------------------------------------------
try:
    # At any nonzero timestep, noisy_latents should differ from clean_latents
    diff = (noisy_latents - clean_latents).abs().sum().item()
    assert diff > 0, "Noisy latents are identical to clean latents --no noise added"

    logger.info(
        f"[PASSED] sample_noise noise added | diff={diff:.6f}"
    )

except AssertionError as e:
    logger.error(f"[FAILED] sample_noise noise added | Error: {e}")
    sys.exit(1)

except Exception as e:
    logger.error(f"[FAILED] sample_noise noise added | Unexpected Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# Noise magnitude increases with each timestep
# -------------------------------------------------------------
try:
    # At low timesteps, noise should be small; at high timesteps, noise should be large
    # test with single clean latent

    single_latent = torch.randn(1, latent_channels, latent_size, latent_size)
    fixed_noise = torch.randn_like(single_latent)

    low_t = torch.tensor([10],dtype=torch.long)
    high_t = torch.tensor([900],dtype=torch.long)

    noisy_low = scheduler.add_noise(single_latent, fixed_noise, low_t)
    noisy_high = scheduler.add_noise(single_latent, fixed_noise, high_t)

    diff_low = (noisy_low - single_latent).abs().mean().item()
    diff_high = (noisy_high - single_latent).abs().mean().item()

    assert diff_low < diff_high, \
        f"Noise magnitude decreases with timestep: diff_low={diff_low:.6f} > diff_high={diff_high:.6f}"

    logger.info(
        f"[PASSED] sample_noise noise magnitude | "
        f"diff_low={diff_low:.6f} | diff_high={diff_high:.6f}"
    )

except AssertionError as e:
    logger.error(f"[FAILED] sample_noise noise magnitude | Error: {e}")
    sys.exit(1)

except Exception as e:
    logger.error(f"[FAILED] sample_noise noise magnitude | Unexpected Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# add_noise() is deterministic 
# -------------------------------------------------------------
try:
    # Same noise + same timestep + same latent should produce same noisy latent
    noisy_a = scheduler.add_noise(single_latent, fixed_noise, low_t)
    noisy_b = scheduler.add_noise(single_latent, fixed_noise, low_t)

    assert torch.equal(noisy_a, noisy_b), \
        "add_noise() is not deterministic -- same noise+timestep should produce same result"

    logger.info("[PASSED] sample_noise add_noise deterministic")

except AssertionError as e:
    logger.error(f"[FAILED] sample_noise add_noise deterministic | Error: {e}")
    sys.exit(1)

except Exception as e:
    logger.error(f"[FAILED] sample_noise add_noise deterministic | Unexpected Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------
logger.info("All tests passed successfully!")
sys.exit(0)