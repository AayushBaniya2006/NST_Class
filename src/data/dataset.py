"""PyTorch Dataset for Fitzpatrick17k skin tone classification."""
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FitzpatrickDataset(Dataset):
    """Dataset for loading Fitzpatrick17k images with skin tone labels.

    Args:
        df: DataFrame with 'hasher' and 'skin_tone_label' columns.
        image_dir: Directory containing the images.
        transform: Optional torchvision transforms to apply.
        label_column: Column name for the integer label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        label_column: str = "skin_tone_label",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, hasher: str) -> Optional[Path]:
        """Find image file by hasher name, trying common extensions."""
        # Check bare hasher path first (handles filenames that already have extensions)
        direct = self.image_dir / hasher
        if direct.is_file():
            return direct
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            path = self.image_dir / f"{hasher}{ext}"
            if path.exists():
                return path
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hasher = row["hasher"]
        label = int(row[self.label_column])

        img_path = self._find_image(hasher)
        if img_path is None:
            raise FileNotFoundError(f"No image found for hasher: {hasher}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
