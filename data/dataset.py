"""
data/dataset.py

Data Pipeline for the Diffusion Model
Streams anime-captions dataset none-yet/anime-captions dataset from huggingface
preprocesses images and tokenizes captions into model-ready batches

- Iterable dataset because HF dataset does not support random access
- Image normalized to  [-1,1] as required by the VAE
- Tokenization happens here (CPU), latent encoding happens in training loop (GPU)
- num_workers = 0 on DataLoader to avoid multiprocessing bugs with streaming dataset

Usage:
    from data.dataset import build_dataloader
    dataloader = build_dataloader(samples_to_skip=0)
    for batch in dataloader:
        pixel_values = batch["pixel_values"] # [batch_size, num_channels, height, width] = [batch_size, 3, 256, 256]
        input_ids = batch["input_ids"] # [batch_size, max_length] = [batch_size, 77]
        attention_mask = batch["attention_mask"] # [batch_size, max_length] = [batch_size, 77
"""

import os
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from transformers import CLIPTokenizer
from datasets import load_dataset
from PIL import Image
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# Image Transforms
# ------------------------------------------------------------


def build_image_transform(image_size: int) -> transforms.Compose:
    """
    Resize to image_size x image_size, center crop, convert to tensor and normalize to [-1,1]

    Center crop after resize preserves aspect ratio better than direct resize-to-square,
    which would distort the image.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),  # only augmentation we use
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # shifts [0,1] to [-1,1]
        ]
    )


# ------------------------------------------------------------
# IterableDataset
# ------------------------------------------------------------


class AnimeStreamDataset(IterableDataset):
    """
    Streams anime-captions from HuggingFace and yields preprocessed samples.

    Args:
        samples_to_skip: Numbers of samples already seen. Used to resume training
                        from a checkpoint by fast-forwarding the stream.
        max_samples:     Cap on maximum samples to yield. Defaults to DATASET_TRAIN_SAMPLES from environment.
    """

    def __init__(self, samples_to_skip: int = 0, max_samples: int = None):
        super().__init__()

        self.dataset_id = os.environ["HF_DATASET_ID"]
        self.dataset_split = os.environ["HF_DATASET_SPLIT"]
        self.shuffle_buffer = int(os.environ["STREAMING_SHUFFLE_BUFFER"])
        self.shuffle_seed = int(os.environ["STREAMING_SHUFFLE_SEED"])
        self.image_size = int(os.environ["IMAGE_SIZE"])
        self.tokenizer_id = os.environ["TOKENIZER_MODEL_ID"]
        self.max_length = int(os.environ["TOKENIZER_MAX_LENGTH"])
        self.max_samples = max_samples or int(os.environ["DATASET_TRAIN_SAMPLES"])
        self.samples_to_skip = samples_to_skip

        self.transform = build_image_transform(self.image_size)

        # Tokenizer is lightweight, so we can keep it in memory, safe to instantiate here
        logger.info(f"Loading tokenizer: {self.tokenizer_id}")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_id)

    def _load_stream(self):
        """Load and configure the HuggingFace streaming dataset."""
        logger.info(
            f"Loading streaming dataset: {self.dataset_id}/[{self.dataset_split}]"
        )
        ds = load_dataset(
            self.dataset_id,
            split=self.dataset_split,
            streaming=True,
        )

        ds = ds.shuffle(seed=self.shuffle_seed, buffer_size=self.shuffle_buffer)

        if self.samples_to_skip > 0:
            logger.info(f"Skipping first {self.samples_to_skip} samples")
            ds = ds.skip(self.samples_to_skip)
            logger.info("Skip Complete. Resuming stream...")

        return ds

    def _preprocess_image(self, image) -> torch.tensor:
        """
        Convert raw dataset image to normalized tensor.
        HuggingFace images are PIL.Image
        """

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Ensure RGB - some images maybe RGBA or grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def _tokenize(self, caption: str) -> dict:
        """
        Tokenize caption string with CLIP tokenizer.
        Truncates at max_length (77), pads short captions.
        Returns dict with input_ids and attention_mask tensors.
        """

        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens.input_ids.squeeze(0),  # (max_length,)
            "attention_mask": tokens.attention_mask.squeeze(0),  # (max_length,)
        }

    def __iter__(self):
        ds = self._load_stream()
        samples_yielded = 0

        for sample in ds:
            if samples_yielded >= self.max_samples:
                break

            try:
                pixel_values = self._preprocess_image(sample["image"])
                tokens = self._tokenize(sample["text"])
            except Exception as e:
                # Skip corrupted samples, log and continue
                logger.warning(f"Skipping corrupted sample: {e}")
                continue

            yield {
                "pixel_values": pixel_values,  # (3, H, W)
                "input_ids": tokens["input_ids"],  # (max_length,)
                "attention_mask": tokens["attention_mask"],  # (max_length,)
            }

            samples_yielded += 1

        logger.info(f"Dataset exhausted after {samples_yielded} samples")


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------


def build_dataloader(
    samples_to_skip: int = 0,
    max_samples: int = None,
) -> DataLoader:
    """
    Build and return the DataLoader for training

    Args:
        samples_to_skip: Numbers of samples already seen. Used to resume training
        max_samples:     Cap on maximum samples to yield. Defaults to DATASET_TRAIN_SAMPLES
    """
    batch_size = int(os.environ["BATCH_SIZE_PER_GPU"])

    dataset = AnimeStreamDataset(
        samples_to_skip=samples_to_skip,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # avoid multiprocessing bugs with streaming dataset
        pin_memory=True,  # speeds up CPU->GPU transfer
    )

    logger.info(
        f"Dataloader ready | batch_size: {batch_size} | "
        f"skip={samples_to_skip} | max_samples={max_samples} or env default"
    )

    return dataloader
