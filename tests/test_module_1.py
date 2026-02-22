"""
tests/test_module_1.py

Verifies the data-pipeline works correctly.

Tests:
1. Dataset initializes without errors.
2. Image transform produces correct shape and value range.
3. Tokenizer produces correct shape.
4. Dataloader yields correctly shaped batches.
5. Resume via samples_to_skip works.

We use max_samples=5 so the test finishes quickly without
streaming thousands of samples.

Run:
    sbatch tests/test_module_1.slurm
"""

import os
import sys
import torch

# Logger
from utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Starting test_module_1")

# -----------------------------------------------------------------------------
# Dataset Initialization
# -----------------------------------------------------------------------------

try:
    from data.dataset import AnimeStreamDataset, build_dataloader
    dataset = AnimeStreamDataset(samples_to_skip=0, max_samples=5)
    logger.info("[PASSED] Dataset initialized successfully")
except Exception as e:
    logger.error(f"[FAILED] Dataset initialization failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Single Sample Inspection
# -----------------------------------------------------------------------------

try:
    sample = next(iter(dataset))
    pv = sample['pixel_values']
    ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    
    expected_image_size = int(os.environ['IMAGE_SIZE'])
    expected_seq_len = int(os.environ['TOKENIZER_MAX_LENGTH'])

    assert pv.shape == torch.Size([3,expected_image_size,expected_image_size]), \
        f"[FAILED] Image shape mismatch: {pv.shape} != {torch.Size([3,expected_image_size,expected_image_size])}"

    assert ids.shape == torch.Size([expected_seq_len]), \
        f"[FAILED] Input IDs shape mismatch: {ids.shape} != {torch.Size([expected_seq_len])}"

    assert attention_mask.shape == torch.Size([expected_seq_len]), \
        f"[FAILED] Attention mask shape mismatch: {attention_mask.shape} != {torch.Size([expected_seq_len])}"

    pv_min, pv_max = pv.min().item(), pv.max().item()
    
    assert pv_min >= -1 and pv_max <= 1, \
        f"[FAILED] pixel values out of [-1,1] range: min={pv_min:.3f}, max={pv_max:.3f}"
    logger.info(f"pixel values: shape={list(pv.shape)}, min={pv_min:.3f}, max={pv_max:.3f}")
    logger.info(f"input IDs: shape={list(ids.shape)}, dtype={ids.dtype}")
    logger.info(f"attention mask: shape={list(attention_mask.shape)}, dtype={attention_mask.dtype}")

    logger.info("[PASSED] Single sample inspection passed")
except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Dataloader Inspection
# -----------------------------------------------------------------------------

try:
    from torch.utils.data import DataLoader
    small_dataset = AnimeStreamDataset(samples_to_skip=0, max_samples=5)
    dataloader = DataLoader(small_dataset, batch_size=2, num_workers=0)

    batch = next(iter(dataloader))
    
    B = batch['pixel_values'].shape[0]
    expected_image_size = int(os.environ['IMAGE_SIZE'])
    expected_seq_len = int(os.environ['TOKENIZER_MAX_LENGTH'])
    

    assert batch['pixel_values'].shape == torch.Size([B, 3, expected_image_size, expected_image_size])
    assert batch['input_ids'].shape == torch.Size([B, expected_seq_len])
    assert batch['attention_mask'].shape == torch.Size([B, expected_seq_len])

    logger.info(f"Batch size: {B}")
    logger.info(f"pixel values: {list(batch['pixel_values'].shape)}")
    logger.info(f"input IDs: {list(batch['input_ids'].shape)}")
    logger.info(f"attention mask: {list(batch['attention_mask'].shape)}")

    logger.info("[PASSED] Dataloader inspection passed")
except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Resume via samples_to_skip
# -----------------------------------------------------------------------------

try:
    ds_from_start = AnimeStreamDataset(samples_to_skip=0, max_samples=3)
    ds_from_skip = AnimeStreamDataset(samples_to_skip=2, max_samples=3)

    batch_from_start = list(iter(ds_from_start))
    batch_from_skip = list(iter(ds_from_skip))

    # Both should yield valid values
    assert len(batch_from_start) == 3, f"Expected 3 samples, got {len(batch_from_start)}"
    assert len(batch_from_skip) == 3, f"Expected 3 samples, got {len(batch_from_skip)}"

    # First sample of skip should differ from first sample of start
    first_start = batch_from_start[0]['input_ids']
    first_skip = batch_from_skip[0]['input_ids']

    are_different = not torch.equal(first_start, first_skip)

    if are_different:
        logger.info("Skip produced different first sample - working as expected")
    else:
        logger.error("Skip produced same first sample - unexpected behavior")
        sys.exit(1)

    logger.info("[PASSED] Resume via samples_to_skip passed")
except AssertionError as e:
    logger.error(f"[FAILED] Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FAILED] Exception: {e}")
    sys.exit(1)

logger.info("All tests passed successfully")
sys.exit(0)