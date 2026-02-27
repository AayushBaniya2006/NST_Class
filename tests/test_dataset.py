import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from src.data.dataset import FitzpatrickDataset


@pytest.fixture
def sample_data(tmp_path):
    """Create sample images and CSV for testing."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(image_dir / f"img_{i}.jpg")
        rows.append({
            "hasher": f"img_{i}",
            "fitzpatrick": (i % 6) + 1,
            "skin_tone_group": ["12", "12", "34", "34", "56", "56"][i % 6],
            "skin_tone_label": [0, 0, 1, 1, 2, 2][i % 6],
            "label": "acne",
        })

    df = pd.DataFrame(rows)
    return str(image_dir), df


class TestFitzpatrickDataset:
    def test_length(self, sample_data):
        image_dir, df = sample_data
        dataset = FitzpatrickDataset(df, image_dir)
        assert len(dataset) == 10

    def test_returns_image_and_label(self, sample_data):
        image_dir, df = sample_data
        dataset = FitzpatrickDataset(df, image_dir)
        image, label = dataset[0]
        assert image is not None
        assert isinstance(label, int)
        assert label in [0, 1, 2]

    def test_with_transforms(self, sample_data):
        from src.data.transforms import get_eval_transforms
        image_dir, df = sample_data
        transform = get_eval_transforms(224)
        dataset = FitzpatrickDataset(df, image_dir, transform=transform)
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)

    def test_missing_image_raises(self, sample_data):
        image_dir, df = sample_data
        df_bad = df.copy()
        df_bad.iloc[0, df_bad.columns.get_loc("hasher")] = "nonexistent"
        dataset = FitzpatrickDataset(df_bad, image_dir)
        with pytest.raises(FileNotFoundError):
            dataset[0]
