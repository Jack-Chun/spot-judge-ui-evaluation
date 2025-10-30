"""
WiserUI-Bench Dataset Loader

Loads and processes the WiserUI-Bench dataset from Hugging Face.
Handles image downloading, metadata extraction, and task descriptions.

Dataset: jeochris/WiserUI-Bench
Reference: https://arxiv.org/html/2505.05026v3
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

from datasets import load_dataset
from PIL import Image
import yaml


class WiserUIBenchLoader:
    """
    Loader for WiserUI-Bench dataset.

    Handles:
    - Dataset downloading from Hugging Face
    - Image caching
    - Metadata extraction
    - Ground truth labels
    """

    def __init__(
        self,
        cache_dir: str = "./cache/datasets",
        config_path: str = "config/experiment_config.yaml"
    ):
        """
        Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded data
            config_path: Path to experiment configuration
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        dataset_config = config.get('dataset', {})
        self.dataset_name = dataset_config.get('source', 'jeochris/WiserUI-Bench')

        # Image processing config
        image_config = dataset_config.get('image', {})
        self.max_image_size = image_config.get('max_size', 1120)
        self.image_format = image_config.get('format', 'PNG')

        # Dataset splits
        self.splits = dataset_config.get('splits', ['test'])

        self.dataset = None
        self.image_cache_dir = self.cache_dir / "images"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, split: str = "test") -> Dict[str, Any]:
        """
        Load WiserUI-Bench dataset from Hugging Face.

        Args:
            split: Dataset split to load ('test', 'train', etc.)

        Returns:
            Dataset dictionary
        """
        print(f"Loading WiserUI-Bench dataset (split: {split})...")

        try:
            self.dataset = load_dataset(
                self.dataset_name,
                split=split,
                cache_dir=str(self.cache_dir)
            )
            print(f"Loaded {len(self.dataset)} samples from {self.dataset_name}")

            return self.dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}") from e

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - id: Sample ID
            - task_description: UI task description (from rationale)
            - image_win_path: Path to winning UI variant
            - image_lose_path: Path to losing UI variant
            - winner: Always 'win' (for compatibility)
            - metadata: Additional metadata
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        sample = self.dataset[idx]

        # WiserUI-Bench uses 'win' and 'lose' field names
        image_win_path = self._cache_image(sample, 'win', idx)
        image_lose_path = self._cache_image(sample, 'lose', idx)

        # Extract task description from rationale
        rationale_list = sample.get('rationale', [])
        task_description = self._format_task_description(sample, rationale_list)

        # Format rationale for display
        rationale_text = self._format_rationale(rationale_list)

        return {
            "id": f"sample_{idx}",
            "task_description": task_description,
            "image_win_path": str(image_win_path),
            "image_lose_path": str(image_lose_path),
            "winner": "win",  # Always 'win' in this dataset
            "metadata": {
                "source": sample.get('source', ''),
                "company": sample.get('company', ''),
                "page_type": sample.get('page_type', ''),
                "industry_domain": sample.get('industry_domain', ''),
                "web_mobile": sample.get('web_mobile', ''),
                "ui_change": sample.get('ui_change', []),
                "rationale": rationale_text,
                "rationale_raw": rationale_list
            }
        }

    def _format_task_description(self, sample: Dict[str, Any], rationale_list: list) -> str:
        """Format task description from sample metadata."""
        # Build task description from available fields
        page_type = sample.get('page_type', 'page')
        company = sample.get('company', 'website')
        ui_changes = sample.get('ui_change', [])

        if ui_changes:
            change_desc = ", ".join(ui_changes)
            task = f"Evaluate two variants of a {company} {page_type} with changes to: {change_desc}"
        else:
            task = f"Evaluate two variants of a {company} {page_type}"

        return task

    def _format_rationale(self, rationale_list: list) -> str:
        """Format rationale list into readable text."""
        if not rationale_list:
            return ""

        parts = []
        for item in rationale_list:
            law_info = item.get('law', {})
            law_name = law_info.get('name', 'UX Principle')
            law_type = law_info.get('type', '')
            reason = item.get('reason', '')

            if law_type:
                parts.append(f"{law_name} ({law_type}): {reason}")
            else:
                parts.append(f"{law_name}: {reason}")

        return " | ".join(parts)

    def _cache_image(
        self,
        sample: Dict[str, Any],
        image_key: str,
        idx: int
    ) -> Path:
        """
        Cache image to local filesystem.

        Args:
            sample: Dataset sample
            image_key: Key for image in sample ('win' or 'lose')
            idx: Sample index

        Returns:
            Path to cached image
        """
        # Generate cache filename - use image_key directly (win/lose)
        cache_filename = f"sample_{idx}_{image_key}.{self.image_format.lower()}"
        cache_path = self.image_cache_dir / cache_filename

        # Check if already cached
        if cache_path.exists():
            return cache_path

        # Get image from sample
        image = sample.get(image_key)

        if image is None:
            raise ValueError(f"Image not found: {image_key} in sample {idx}")

        # Handle different image formats
        if isinstance(image, Image.Image):
            pil_image = image
        elif hasattr(image, 'convert'):  # datasets.Image object
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Resize if needed
        if max(pil_image.size) > self.max_image_size:
            pil_image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)

        # Save to cache
        pil_image.save(cache_path, format=self.image_format)

        return cache_path

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """
        Get all samples from the dataset.

        Returns:
            List of sample dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        return [self.get_sample(i) for i in range(len(self.dataset))]

    def get_subset(
        self,
        num_samples: Optional[int] = None,
        category: Optional[str] = None,
        start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get a subset of samples.

        Args:
            num_samples: Number of samples to retrieve (None = all)
            category: Filter by category (optional)
            start_idx: Starting index

        Returns:
            List of sample dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        all_samples = self.get_all_samples()

        # Filter by category if specified
        if category:
            all_samples = [
                s for s in all_samples
                if s['metadata'].get('category') == category
            ]

        # Apply start index and limit
        end_idx = start_idx + num_samples if num_samples else len(all_samples)
        return all_samples[start_idx:end_idx]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        all_samples = self.get_all_samples()

        # Count by category
        categories = {}
        for sample in all_samples:
            cat = sample['metadata'].get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        # Count by winner
        winners = {'a': 0, 'b': 0, 'unknown': 0}
        for sample in all_samples:
            winner = sample.get('winner', 'unknown')
            winners[winner] = winners.get(winner, 0) + 1

        return {
            "total_samples": len(all_samples),
            "categories": categories,
            "winner_distribution": winners,
            "splits": self.splits
        }

    def export_sample_list(self, output_path: str):
        """
        Export sample list for inspection.

        Args:
            output_path: Path to output JSON file
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        samples = self.get_all_samples()

        # Simplify for export (exclude large fields)
        export_data = [
            {
                "id": s['id'],
                "task": s['task_description'][:100] + "..." if len(s['task_description']) > 100 else s['task_description'],
                "winner": s['winner'],
                "category": s['metadata'].get('category', ''),
            }
            for s in samples
        ]

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported {len(export_data)} samples to {output_path}")


# Utility function for quick dataset exploration
def explore_dataset():
    """Quick exploration script."""
    loader = WiserUIBenchLoader()
    loader.load()

    stats = loader.get_statistics()
    print("\n=== WiserUI-Bench Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nCategories:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")
    print(f"\nWinner distribution:")
    for winner, count in stats['winner_distribution'].items():
        print(f"  {winner}: {count}")

    # Show first sample
    sample = loader.get_sample(0)
    print("\n=== First Sample ===")
    print(f"ID: {sample['id']}")
    print(f"Task: {sample['task_description']}")
    print(f"Winner: {sample['winner']}")
    print(f"Image A: {sample['image_a_path']}")
    print(f"Image B: {sample['image_b_path']}")


if __name__ == "__main__":
    explore_dataset()
