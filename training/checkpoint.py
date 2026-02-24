"""
training/checkpoint.py

Checkpoinnt management for training state.

Handles:
    - Saving full training state ( UNet, optimizer, LR scheduler, metadata)
    - Loading/resuming from existing checkpoints
    - Rotating old checkpoints (keep last N)
    - Saving best checkpoints seperately

The checkpoint contains everything needed to resume training exactly where it left off,
including number of samples seen for streaming data.

Checkpoints directory structure:
    checkpoints/
        step_500/           # Accelerate state directory
            ...
        step_1000/
            ...
        step_1500/
            ...
        best/               # Best checkpoint (by lowest loss)
            ...
        metadata.json        # Global metadata (step, samples_seen, best_loss)

Usage:
    from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
"""

import os
import json
import shutil

from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# Metadata Management
# ------------------------------------------------------------


def _metadata_path(checkpoint_dir: str) -> str:
    """Return path to the global metadata JSON file."""
    return os.path.join(checkpoint_dir, "metadata.json")


def save_metadata(
    checkpoint_dir: str,
    global_step: int,
    samples_seen: int,
    epoch: int,
    best_loss: float,
    latest_checkpoint_name: str,
):
    """
    Save training metadata to a JSON file.

    This metadata is seperate from accelerate's state and stores information
    needed for resume logic:
        - global_step: which training step we are one
        - samples_seen: how many samples the dataloader has yielded
                        (used to skip forward in the streaming dataset)
        - epoch: current epoch number
        - best_loss: lowest average loss seen so far
        - latest_checkpoint_name: name of the most recent checkpoint directory

    Args:
        checkpoint_dir: Root checkpoint directory
        global_step: Current global training step
        samples_seen: Total samples processed so far
        epoch: Current epoch
        best_loss: Best (lowest) average loss observed
        latest_checkpoint_name: Name of the most recent checkpoint directory
    """
    metadata = {
        "global_step": global_step,
        "samples_seen": samples_seen,
        "epoch": epoch,
        "best_loss": best_loss,
        "latest_checkpoint_name": latest_checkpoint_name,
    }

    path = _metadata_path(checkpoint_dir)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(
        f"Metadata saved | step: {global_step} | samples_seen: {samples_seen} | epoch: {epoch} | best_loss: {best_loss} | latest_checkpoint: {latest_checkpoint_name}"
    )


def load_metadata(checkpoint_dir: str) -> dict:
    """
    Load training metadata from a JSON file.

    Returns:
        dict: global_step, samples_seen, epoch, best_loss, latest_checkpoint_name
        Returns None if metadata file doesn't exist
    """
    path = _metadata_path(checkpoint_dir)
    if not os.path.exists(path):
        logger.warning(f"No metadata file found -- Fresh Run")
        return None

    with open(path, "r") as f:
        metadata = json.load(f)

    logger.info(
        f"Metadata loaded | step: {metadata['global_step']} | samples_seen: {metadata['samples_seen']} | "
        f"epoch: {metadata['epoch']} | best_loss: {metadata['best_loss']} | latest_checkpoint: {metadata['latest_checkpoint_name']}"
    )
    return metadata


# ------------------------------------------------------------
# Checkpoint Save
# ------------------------------------------------------------


def save_checkpoint(
    accelerator,
    checkpoint_dir: str,
    global_step: int,
    samples_seen: int,
    epoch: int,
    best_loss: float,
    current_loss: float = None,
):
    """
    Save full training checkpoint.

    Uses accelerate's save_state() which saves:
        - Model state dict (UNet weights)
        - Optimizer state dict
        - LR scheduler state dict
        - Accelerator random states

    Additionally saves our custom metadata and manages
    checkpoint rotation.

    If current_loss is provided and is lower than best _loss,
    saves a seperate 'best' checkpoint.

    Args:
        - accelerator: Accelerate instance
        - checkpoint_dir: Root checkpoint directory
        - global_step: Current training step
        - samples_seen: Total samples processed so far
        - epoch: Current epoch
        - best_loss: Best (lowest) average loss observed
        - current_loss: Loss for current checkpoint interval (optional)

    Returns:
        best_loss: Updated best loss (may be same as input if no improvement)
    """

    # Only rank 0 manages checkpoint logic, but all ranks must call save_state
    checkpoint_name = f"step_{global_step}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # --- Save Accelerate State (all ranks) ---
    accelerator.save_state(checkpoint_path)

    if accelerator.is_main_process:
        # --- Save Metadata ---
        save_metadata(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            samples_seen=samples_seen,
            epoch=epoch,
            best_loss=best_loss,
            latest_checkpoint_name=checkpoint_name,
        )

        # --- Save Best Checkpoint (if improved) ---
        if current_loss is not None and current_loss < best_loss:
            best_loss = current_loss
            best_path = os.path.join(checkpoint_dir, "best")

            # Remove old best checkpoint if it exists
            if os.path.exists(best_path):
                shutil.rmtree(best_path)

            # Copy current checkpoint to best
            shutil.copytree(checkpoint_path, best_path)

            # Update metadata with new best loss
            save_metadata(
                checkpoint_dir=checkpoint_dir,
                global_step=global_step,
                samples_seen=samples_seen,
                epoch=epoch,
                best_loss=best_loss,
                latest_checkpoint_name=checkpoint_name,
            )
            logger.info(
                f"New best checkpoint saved | step: {global_step} | loss: {best_loss:.6f}"
            )

        # --- Rotate Checkpoints ---
        _rotate_checkpoints(checkpoint_dir)

    # --- Ensure all ranks wait for rank 0 to complete ---
    accelerator.wait_for_everyone()

    logger.info(f"Checkpoint saved | step={global_step} | path={checkpoint_path}")
    return best_loss


# ------------------------------------------------------------
# Checkpoint Load/Resume
# ------------------------------------------------------------


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Find hte path to latest checkpoint directory.

    Reads from metadata.json to determine which checkpoint to load.

    Args:
        checkpoint_dir: Root checkpoint directory

    Returns:
        Full path to the latest checkpoint directory, or None if not found.
    """

    metadata = load_metadata(checkpoint_dir)
    if metadata is None:
        return None

    latest_name = metadata.get("latest_checkpoint_name")
    if latest_name is None:
        return None

    latest_path = os.path.join(checkpoint_dir, latest_name)
    if not os.path.exists(latest_path):
        logger.warning(
            f"Metadata points to non-existent checkpoint: {latest_path}"
            "Starting Fresh Run"
        )
        return None

    return latest_path


def load_checkpoint(accelerator, checkpoint_dir: str) -> dict | None:
    """
    Resume training from the latest checkpoint.

    Loads accelerate state (model, optimizer, LR scheduler, random states)
    and returns the metadata dict so the training loop can restore
    global_step and samples_seen.

    Args:
        accelerator: Accelerate instance
        checkpoint_dir: Root checkpoint directory

    Returns:
        metadata dict if checkpoint was loaded, None if starting fresh
    """

    resume_from = os.environ.get("RESUME_FROM_CHECKPOINT", "latest")

    if resume_from.lower() == "none":
        logger.info("RESUME_FROM_CHECKPOINT is set to None -- Starting Fresh Run")
        return None

    latest_path = find_latest_checkpoint(checkpoint_dir)
    if latest_path is None:
        logger.info("No checkpoints found -- Starting Fresh Run")
        return None

    logger.info(f"Resuming from checkpoint | path={latest_path}")
    accelerator.load_state(latest_path)

    metadata = load_metadata(checkpoint_dir)
    logger.info(
        f"Checkpoint loaded | step={metadata['global_step']} | "
        f"samples_seen={metadata['samples_seen']} | epoch={metadata['epoch']} | "
        f"best_loss={metadata['best_loss']:.6f}"
    )
    return metadata


# ------------------------------------------------------------
# Checkpoint Rotation
# ------------------------------------------------------------


def _rotate_checkpoints(checkpoint_dir: str):
    """
    Keep only the last N checkpoints, delete older ones.

    Reads KEEP_LAST_N_CHECKPOINTS (default 3).
    Never delete the 'best' checkpoint.

    Checkpoints are identified by the naming convention "step_<global_step>"
    and sorted by global_step.
    """

    keep_n = int(os.environ.get("KEEP_LAST_N_CHECKPOINTS", 3))

    # Find all step_* directories
    checkpoint_dirs = []
    for name in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(full_path) and name.startswith("step_"):
            try:
                step_num = int(name.split("_")[1])
                checkpoint_dirs.append((step_num, full_path))
            except (ValueError, IndexError):
                continue

    # Sort by global_step ascending
    checkpoint_dirs.sort(key=lambda x: x[0])

    # Delete oldest checkpoint beyond keep limit
    if len(checkpoint_dirs) > keep_n:
        to_delete = checkpoint_dirs[: len(checkpoint_dirs) - keep_n]
        for _, path in to_delete:
            logger.info(f"Rotating checkpoint | path={path}")
            shutil.rmtree(path)
