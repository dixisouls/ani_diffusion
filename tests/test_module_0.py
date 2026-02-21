"""
tests/test_module_0.py

Verfies:
1. All required env variables are set
2. Logger initializes correctly
3. Looger writes to stdout and file without errors
"""

import os
import sys



# ------------------------------------------------------------
# 1. Logger Checks  
# ------------------------------------------------------------

try:
    from utils.logger import get_logger
    logger = get_logger("test_module0", local_rank=0)
    logger.info("Logger initialized successfully.")
    logger.debug("This is a DEBUG message (only visible if LOG_LEVEL=DEBUG).")
    logger.warning("This is a WARNING message.")

    # Verify log file was created
    log_file = os.path.join(os.environ["LOG_DIR"], "train.log")
    if os.path.exists(log_file):
        print(f"\n  OK        Log file created at: {log_file}")
    else:
        print(f"\n  WARNING   Log file not found at: {log_file}")
        print("            This may be a timing issue with enqueue=True.")
        print("            Check the file manually after the script exits.")

    print("\nPASSED: Logger initialized and wrote messages without errors.")

except ImportError as e:
    print(f"\nFAILED: Could not import logger -- {e}")
    print("        Make sure you are running from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"\nFAILED: Logger raised an unexpected error -- {e}")
    sys.exit(1)

# ------------------------------------------------------------
# 2. Environment variables checks
# ------------------------------------------------------------

REQUIRED_ENV_VARS = [
    # Paths
    "PROJECT_ROOT",
    "CHECKPOINT_DIR",
    "LOG_DIR",
    "OUTPUT_DIR",
    "WANDB_DIR",
    # HuggingFace
    "HF_DATASET_ID",
    "HF_DATASET_SPLIT",
    "VAE_MODEL_ID",
    "TEXT_ENCODER_MODEL_ID",
    "TOKENIZER_MODEL_ID",
    "UNET_MODEL_ID",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
    # Dataset
    "DATASET_TOTAL_SAMPLES",
    "DATASET_TRAIN_SAMPLES",
    "STREAMING_SHUFFLE_BUFFER",
    "STREAMING_SHUFFLE_SEED",
    # Image
    "IMAGE_SIZE",
    "LATENT_SIZE",
    "LATENT_CHANNELS",
    # Text
    "TOKENIZER_MAX_LENGTH",
    # Training
    "LEARNING_RATE",
    "LR_WARMUP_STEPS",
    "LR_SCHEDULER_TYPE",
    "OPTIMIZER_TYPE",
    "WEIGHT_DECAY",
    "GRAD_CLIP",
    "BATCH_SIZE_PER_GPU",
    "NUM_GPUS",
    "MIXED_PRECISION",
    # Scheduler
    "SCHEDULER_TYPE",
    "NUM_TRAIN_TIMESTEPS",
    "BETA_SCHEDULE",
    "INFERENCE_SCHEDULER_TYPE",
    "INFERENCE_NUM_STEPS",
    # Checkpointing
    "CHECKPOINT_EVERY_N_STEPS",
    "KEEP_LAST_N_CHECKPOINTS",
    "RESUME_FROM_CHECKPOINT",
    # EMA
    "USE_EMA",
    "EMA_DECAY",
    # Logging
    "LOG_LEVEL",
    # WandB
    "WANDB_PROJECT",
    "WANDB_MODE",
    # Max steps
    "MAX_TRAIN_STEPS",
]

missing_env_vars = []
for var in REQUIRED_ENV_VARS:
    value = os.environ.get(var)
    if value is None:
        missing_env_vars.append(var)
        logger.warning(f"Environment variable {var} is not set.")
    else:
        logger.info(f"Environment variable {var} is set to: {value}")

if missing_env_vars:
    logger.error(f"Missing environment variables: {missing_env_vars}")
    sys.exit(1)
else:
    logger.info("All environment variables are set.")

# ------------------------------------------------------------
# 3. Directory creation checks
# ------------------------------------------------------------

dirs_to_create = [
    os.environ["CHECKPOINT_DIR"],
    os.environ["LOG_DIR"],
    os.environ["OUTPUT_DIR"],
    os.environ["WANDB_DIR"],
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"]
]

dir_failed = False
for d in dirs_to_create:
    try:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory {d} created successfully.")
    except Exception as e:
        logger.error(f"Failed to create directory {d} -- {e}")
        dir_failed = True

if dir_failed:
    logger.error("Failed to create one or more directories.")
    sys.exit(1)
else:
    logger.info("All directories created successfully.")


# ------------------------------------------------------------
logger.info("All tests passed successfully.")
sys.exit(0)