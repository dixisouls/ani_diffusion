# Anime Diffusion

A latent diffusion model fine-tuned for anime image generation, built on top of Stable Diffusion. Supports distributed training with EMA, classifier-free guidance inference, and full checkpoint management.

## Features

- **Latent diffusion** — fine-tunes a UNet on top of frozen VAE and CLIP encoders from Stable Diffusion
- **Distributed training** — multi-GPU support via HuggingFace Accelerate
- **EMA** — exponential moving average of UNet weights for higher-quality outputs
- **Classifier-free guidance (CFG)** — controllable prompt adherence at inference
- **Streaming dataset** — streams directly from HuggingFace Hub, no local disk storage needed
- **Checkpoint management** — auto-resume, checkpoint rotation, and best-model tracking
- **WandB logging** — offline-mode metric tracking with loss, LR, and step time

## Architecture

```
ani_diffusion/
├── train.py              # Training entry point
├── config.env            # All hyperparameters and paths
├── requirements.txt
├── models/
│   ├── loader.py         # Load VAE, CLIP text encoder, UNet from pretrained
│   └── scheduler.py      # DDPM (training) and DDIM (inference) schedulers
├── training/
│   ├── loop.py           # Single training step
│   ├── checkpoint.py     # Save / load / resume checkpoints
│   └── ema.py            # EMA shadow weight management
├── inference/
│   └── generate.py       # Image generation pipeline with CFG
├── data/
│   └── dataset.py        # Streaming IterableDataset with resume support
└── utils/
    └── logger.py         # Loguru-based structured logging
```

## Setup

**Python 3.12+ and CUDA 12.8 required.**

```bash
pip install -r requirements.txt
```

Key dependencies: `torch 2.10`, `diffusers 0.36`, `transformers 5.2`, `accelerate 1.12`, `datasets 4.5`, `wandb 0.25`.

## Training

Edit `config.env` to set your paths and hyperparameters, then launch:

```bash
source config.env
accelerate launch train.py
```

Training will automatically resume from the latest checkpoint if one exists. Key defaults (configurable in `config.env`):

| Parameter | Default |
|---|---|
| `LEARNING_RATE` | `1e-4` |
| `BATCH_SIZE_PER_GPU` | `32` |
| `NUM_GPUS` | `4` |
| `MAX_TRAIN_STEPS` | `50000` |
| `CHECKPOINT_EVERY_N_STEPS` | `500` |
| `EMA_DECAY` | `0.9995` |
| `MIXED_PRECISION` | `fp16` |

Checkpoints are saved to `./checkpoints/`, logs to `./logs/`, and WandB runs to `./wandb/`.

## Inference

```bash
python inference/generate.py \
  --prompt "anime girl with purple eyes in a magical forest" \
  --checkpoint_dir ./checkpoints \
  --output_dir ./outputs \
  --num_images 4 \
  --num_steps 30 \
  --guidance_scale 7.5 \
  --seed 42
```

Generated images are saved as PNGs to `--output_dir`. The pipeline applies EMA weights automatically if a checkpoint with EMA state is found.

`--guidance_scale` controls prompt adherence: `1.0` = no guidance, `7.5` = standard, higher values increase prompt strictness.

## Checkpoints

Each checkpoint saves the full training state:

```
checkpoints/
  step_500/
    ema_state.pt        # EMA shadow weights
  step_1000/
  best/                 # Lowest-loss checkpoint
  metadata.json         # global_step, samples_seen, epoch, best_loss
```

Only the last 3 checkpoints are kept (configurable via `KEEP_LAST_N_CHECKPOINTS`).

## Dataset

Training uses the [`none-yet/anime-captions`](https://huggingface.co/datasets/none-yet/anime-captions) dataset (~337k image-caption pairs), streamed directly from HuggingFace Hub. Images are resized and center-cropped to 512×512 and normalized to `[-1, 1]`.

## License

See [LICENSE](LICENSE).
