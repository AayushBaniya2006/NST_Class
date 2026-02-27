"""Image augmentation pipelines for train, validation, and test sets."""
from torchvision import transforms


def get_train_transforms(image_size: int = 224, config: dict = None) -> transforms.Compose:
    """Training augmentation pipeline with horizontal flip, rotation, color jitter, and normalization."""
    cfg = config or {}
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg.get("rotation_degrees", 15)),
        transforms.ColorJitter(
            brightness=cfg.get("brightness", 0.2),
            contrast=cfg.get("contrast", 0.2),
            saturation=cfg.get("saturation", 0.2),
            hue=cfg.get("hue", 0.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transform pipeline. No augmentation, just resize and normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
