"""
Test Script for Spotter System and User Prompts

This script helps you experiment with different prompt variations for the Spotter
to improve its performance at identifying UI/UX differences.

Usage:
    python test_spotter_prompts.py --image-a path/to/baseline.png --image-b path/to/variant.png
    python test_spotter_prompts.py --sample-id 0  # Uses WiserUI-Bench sample
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

from src.models.glm4v_spotter import GLM4VSpotter
from src.utils.prompt_templates import SPOTTER_SYSTEM_PROMPT, SPOTTER_USER_PROMPT

import dotenv
dotenv.load_dotenv()


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_single_prompt(
    system_prompt: str,
    user_prompt: str,
    image_a: str,
    image_b: str,
    prompt_name: str = "Custom"
) -> Dict[str, Any]:
    """
    Test a single prompt variation.

    Args:
        system_prompt: System prompt to test
        user_prompt: User prompt to test
        image_a: Path to first image
        image_b: Path to second image
        prompt_name: Name for this prompt variation

    Returns:
        Test result with output and metadata
    """
    print(f"\n{'='*80}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*80}")

    spotter = GLM4VSpotter(
        api_key=os.getenv('ZAI_API_KEY'),
        config_path="config/models_config.yaml"
    )

    print(f"Analyzing images:")
    print(f"  Baseline: {image_a}")
    print(f"  Variant:  {image_b}")
    print(f"\nCalling GLM-4.5V API...")

    try:
        output, metadata = spotter.spot_differences(image_a, image_b)

        print(f"\n✓ Success!")
        print(f"  Latency: {metadata['latency_seconds']:.2f}s")
        print(f"  Tokens: {metadata['total_tokens']} (in: {metadata['input_tokens']}, out: {metadata['output_tokens']})")

        # Display output
        print(f"\nOutput:")
        print(json.dumps(output, indent=2))

        return {
            "prompt_name": prompt_name,
            "success": True,
            "output": output,
            "metadata": metadata,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return {
            "prompt_name": prompt_name,
            "success": False,
            "error": str(e),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }


def load_sample_from_dataset(sample_id: int) -> Tuple[str, str]:
    """Load a sample from WiserUI-Bench dataset."""
    from src.data.dataset_loader import WiserUIBenchLoader

    loader = WiserUIBenchLoader()
    loader.load(split='test')
    sample = loader.get_sample(sample_id)

    return sample['image_lose_path'], sample['image_win_path']


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test and compare Spotter prompt variations"
    )

    # Input options
    parser.add_argument(
        '--image-a',
        type=str,
        help='Path to baseline image'
    )
    parser.add_argument(
        '--image-b',
        type=str,
        help='Path to variant image'
    )
    parser.add_argument(
        '--sample-id',
        type=int,
        help='Use WiserUI-Bench sample by ID'
    )

    args = parser.parse_args()

    # Get images
    if args.sample_id is not None:
        print(f"Loading WiserUI-Bench sample {args.sample_id}...")
        image_a, image_b = load_sample_from_dataset(args.sample_id)
    elif args.image_a and args.image_b:
        image_a = args.image_a
        image_b = args.image_b
    else:
        parser.error("Must provide either --sample-id or both --image-a and --image-b")

    # Verify images exist
    if not Path(image_a).exists():
        print(f"Error: Image not found: {image_a}")
        return
    if not Path(image_b).exists():
        print(f"Error: Image not found: {image_b}")
        return

    test_single_prompt(
        system_prompt=SPOTTER_SYSTEM_PROMPT,
        user_prompt=SPOTTER_USER_PROMPT,
        image_a=image_a,
        image_b=image_b
    )

    test_single_prompt(
        system_prompt=SPOTTER_SYSTEM_PROMPT,
        user_prompt=SPOTTER_USER_PROMPT,
        image_a=image_b,
        image_b=image_a
    )


if __name__ == "__main__":
    main()
