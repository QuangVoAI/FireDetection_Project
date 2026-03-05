"""
Tests for the data preprocessing pipeline.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Allow project-root imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import (
    collect_image_paths,
    compute_hash,
    deduplicate,
    resize_image,
    split_dataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """
    Create a minimal fake data root with images in source folder 01.
    Returns the path to the data root.
    """
    img_dir = tmp_path / "01_Positive_Standard" / "images"
    img_dir.mkdir(parents=True)
    for i in range(5):
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        img.save(img_dir / f"image_{i:03d}.jpg")
    return tmp_path


@pytest.fixture
def tmp_data_root_with_labels(tmp_path: Path) -> Path:
    """Data root with matching YOLO labels."""
    img_dir = tmp_path / "01_Positive_Standard" / "images"
    lbl_dir = tmp_path / "01_Positive_Standard" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        )
        img.save(img_dir / f"img_{i}.jpg")
        (lbl_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    return tmp_path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCollectImagePaths:
    def test_collects_images_from_folder(self, tmp_data_root):
        paths = collect_image_paths(tmp_data_root)
        assert len(paths) == 5

    def test_returns_empty_list_for_empty_root(self, tmp_path):
        paths = collect_image_paths(tmp_path)
        assert paths == []

    def test_only_includes_image_files(self, tmp_data_root):
        # Add a non-image file – it must be excluded
        (tmp_data_root / "01_Positive_Standard" / "images" / "notes.txt").write_text("hi")
        paths = collect_image_paths(tmp_data_root)
        assert all(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for p in paths)


class TestResizeImage:
    def test_output_is_square(self):
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = resize_image(img, size=640)
        assert result.shape == (640, 640, 3)

    def test_square_input(self):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = resize_image(img, size=640)
        assert result.shape == (640, 640, 3)

    def test_preserves_dtype(self):
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = resize_image(img, size=320)
        assert result.dtype == np.uint8


class TestComputeHash:
    def test_same_image_same_hash(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        p = tmp_path / "img.png"
        img.save(p)
        h1 = compute_hash(p)
        h2 = compute_hash(p)
        assert h1 == h2

    def test_different_images_different_hash(self, tmp_path):
        rng = np.random.default_rng(42)
        for i in range(2):
            img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(tmp_path / f"img_{i}.png")
        h1 = compute_hash(tmp_path / "img_0.png")
        h2 = compute_hash(tmp_path / "img_1.png")
        # Random images should produce different perceptual hashes
        assert h1 != h2


class TestDeduplicate:
    def test_removes_exact_duplicates(self, tmp_path):
        img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 128)
        paths = []
        for i in range(4):
            p = tmp_path / f"dup_{i}.jpg"
            img.save(p)
            paths.append(p)
        result = deduplicate(paths, hash_threshold=0)
        assert len(result) == 1

    def test_keeps_unique_images(self, tmp_path):
        paths = []
        # Create 5 visually distinct images using random noise with different seeds
        # Random noise images are near-orthogonal in hash space
        for i in range(5):
            rng = np.random.default_rng(seed=i * 1000 + 42)
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            p = tmp_path / f"unique_{i}.png"
            Image.fromarray(arr).save(p)
            paths.append(p)
        # Use a very tight threshold (0 = exact hash match only)
        result = deduplicate(paths, hash_threshold=0)
        assert len(result) == 5

    def test_handles_empty_list(self):
        result = deduplicate([])
        assert result == []


class TestSplitDataset:
    def test_split_sizes(self):
        paths = [Path(f"img_{i}.jpg") for i in range(100)]
        train, val, test = split_dataset(paths, train_ratio=0.8, val_ratio=0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        paths = [Path(f"img_{i}.jpg") for i in range(50)]
        train, val, test = split_dataset(paths)
        all_sets = set(train) | set(val) | set(test)
        assert len(all_sets) == len(train) + len(val) + len(test)

    def test_reproducible_with_seed(self):
        paths = [Path(f"img_{i}.jpg") for i in range(30)]
        t1, v1, _ = split_dataset(paths, seed=7)
        t2, v2, _ = split_dataset(paths, seed=7)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds_give_different_splits(self):
        paths = [Path(f"img_{i}.jpg") for i in range(30)]
        t1, _, _ = split_dataset(paths, seed=1)
        t2, _, _ = split_dataset(paths, seed=2)
        assert t1 != t2
