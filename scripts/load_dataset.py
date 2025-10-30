"""
Test script for WiserUI-Bench dataset loader.

Tests dataset downloading, caching, and sample retrieval.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import WiserUIBenchLoader


def load_dataset():
    """Test basic dataset loading."""
    print("=" * 80)
    print("Testing WiserUI-Bench Dataset Loader")
    print("=" * 80)

    try:
        # Initialize loader
        print("\n1. Initializing dataset loader...")
        loader = WiserUIBenchLoader()
        print("✓ Loader initialized")

        # Load dataset
        print("\n2. Loading dataset from Hugging Face...")
        print("   (This may take a while on first run)")
        dataset = loader.load(split='test')
        print(f"✓ Dataset loaded: {len(dataset)} samples")

        # Get statistics
        print("\n3. Getting dataset statistics...")
        stats = loader.get_statistics()
        print(f"✓ Total samples: {stats['total_samples']}")
        print("\n   Categories:")
        for cat, count in stats['categories'].items():
            print(f"     - {cat}: {count}")
        print("\n   Winner distribution:")
        for winner, count in stats['winner_distribution'].items():
            print(f"     - {winner}: {count}")

        # Test sample retrieval
        print("\n4. Testing sample retrieval...")
        sample = loader.get_sample(0)
        print(f"✓ Retrieved sample 0")
        print(f"   ID: {sample['id']}")
        print(f"   Task: {sample['task_description'][:100]}...")
        print(f"   Winner: {sample['winner']}")
        print(f"   Company: {sample['metadata']['company']}")
        print(f"   Page type: {sample['metadata']['page_type']}")
        print(f"   Image (win): {sample['image_win_path']}")
        print(f"   Image (lose): {sample['image_lose_path']}")

        # Verify images exist
        print("\n5. Verifying cached images...")
        img_win_path = Path(sample['image_win_path'])
        img_lose_path = Path(sample['image_lose_path'])

        if img_win_path.exists():
            print(f"✓ Image (win) cached: {img_win_path}")
        else:
            print(f"✗ Image (win) not found: {img_win_path}")

        if img_lose_path.exists():
            print(f"✓ Image (lose) cached: {img_lose_path}")
        else:
            print(f"✗ Image (lose) not found: {img_lose_path}")

        # Export sample list
        print("\n6. Exporting sample list...")
        export_path = "cache/sample_list.json"
        loader.export_sample_list(export_path)
        print(f"✓ Sample list exported to: {export_path}")

        print("\n" + "=" * 80)
        print("✓ Dataset loader test completed successfully!")
        print("=" * 80)

        return sample

    except Exception as e:
        print(f"\n✗ Dataset loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    sample = load_dataset()
    sys.exit(0 if sample else 1)
