"""Image augmentation pipelines for train, validation, and test sets.

Supports 5 augmentation buckets + control for ablation experiments:
    control:  No augmentation (resize + normalize only)
    flip:     RandomHorizontalFlip(p=0.5)
    rotate:   RandomRotation(±15°)
    crop:     RandomResizedCrop(scale=(0.9, 1.1))
    noise:    GaussianNoise(p=0.2, mean=0, std=0.03)
    combined: flip + rotate + crop together

References:
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC5977656/
    - https://www.sciencedirect.com/science/article/pii/S0169260721003102
"""
import torch
from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Valid augmentation bucket names
AUGMENTATION_BUCKETS = ["control", "flip", "rotate", "crop", "noise", "combined"]


class GaussianNoise:
    """Add Gaussian noise to a tensor image.

    Applied after ToTensor (values in [0, 1]), so std is in that scale.
    Noise is sampled from N(mean, std) and added to each pixel independently.

    Reference: PMC5977656 — "We generate an array N where each element
    is a sample from a gaussian distribution with μ=0."

    Args:
        p: Probability of applying noise.
        mean: Mean of the Gaussian noise.
        std: Standard deviation (in [0,1] scale since input is normalized to [0,1]).
    """

    def __init__(self, p: float = 0.2, mean: float = 0.0, std: float = 0.03):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, mean={self.mean}, std={self.std})"


def get_augmentation_transforms(bucket: str, image_size: int = 224) -> list:
    """Return the augmentation-specific transforms for a given bucket.

    Args:
        bucket: One of AUGMENTATION_BUCKETS.
        image_size: Target image size.

    Returns:
        List of transforms to insert between Resize and ToTensor.
    """
    if bucket not in AUGMENTATION_BUCKETS:
        raise ValueError(
            f"Unknown augmentation bucket '{bucket}'. "
            f"Valid options: {AUGMENTATION_BUCKETS}"
        )

    if bucket == "control":
        return [], []

    if bucket == "flip":
        return [transforms.RandomHorizontalFlip(p=0.5)], []

    if bucket == "rotate":
        return [transforms.RandomRotation(degrees=15)], []

    if bucket == "crop":
        # RandomResizedCrop replaces Resize — it crops and resizes in one step
        return [transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.9, 1.1),
        )], []

    if bucket == "noise":
        # GaussianNoise operates on tensors, so it goes after ToTensor
        return [], [GaussianNoise(p=0.2, mean=0.0, std=0.03)]

    if bucket == "combined":
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.9, 1.1),
            ),
        ], []

    return [], []


def get_train_transforms(
    image_size: int = 224,
    augmentation: str = "combined",
) -> transforms.Compose:
    """Training transform pipeline with selectable augmentation bucket.

    Args:
        image_size: Target image size (default 224).
        augmentation: Augmentation bucket name (default "combined").
            One of: control, flip, rotate, crop, noise, combined.

    Returns:
        Composed transform pipeline.
    """
    pre_tensor_augs, post_tensor_augs = get_augmentation_transforms(
        augmentation, image_size
    )

    pipeline = []

    # For crop bucket, RandomResizedCrop already handles resize
    has_crop = any(
        isinstance(t, transforms.RandomResizedCrop) for t in pre_tensor_augs
    )
    if not has_crop:
        pipeline.append(transforms.Resize((image_size, image_size)))

    pipeline.extend(pre_tensor_augs)
    pipeline.append(transforms.ToTensor())
    pipeline.extend(post_tensor_augs)
    pipeline.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(pipeline)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transform pipeline. No augmentation, just resize and normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
