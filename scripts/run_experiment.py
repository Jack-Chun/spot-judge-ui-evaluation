"""
Run full Spot & Judge experiment on WiserUI-Bench.

This script runs the complete experiment comparing:
- Spot & Judge (GLM-4.5V → GPT-4o/Claude)
- Baseline E2E (GPT-4o/Claude direct evaluation)

Across all 300 samples with position permutation testing.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.orchestrator import ExperimentOrchestrator
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Spot & Judge experiment')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to test (default: all 300)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['gpt-4o', 'claude-3.5-sonnet'],
        help='Models to test (default: gpt-4o claude-3.5-sonnet)'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline evaluation (only run Spot & Judge)'
    )
    parser.add_argument(
        '--skip-spot-judge',
        action='store_true',
        help='Skip Spot & Judge (only run baseline)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint file'
    )

    args = parser.parse_args()

    # Determine architectures
    architectures = []
    if not args.skip_spot_judge:
        architectures.append('spot_and_judge')
    if not args.skip_baseline:
        architectures.append('baseline_e2e')

    if not architectures:
        print("Error: Cannot skip both architectures!")
        return 1

    # Model configuration
    models = {
        'spot_and_judge': args.models,
        'baseline_e2e': args.models
    }

    print("=" * 80)
    print("SPOT & JUDGE EXPERIMENT")
    print("=" * 80)
    print(f"Samples: {args.num_samples or 'ALL (300)'}")
    print(f"Architectures: {', '.join(architectures)}")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 80)
    print()

    # Confirm if running on full dataset
    if args.num_samples is None and not args.resume:
        response = input("Running on ALL 300 samples. This may take several hours and cost $50-200. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return 0

    # Initialize orchestrator
    print("\nInitializing experiment orchestrator...")
    orchestrator = ExperimentOrchestrator(resume_from_checkpoint=args.resume)

    # Run experiment
    print("\nStarting experiment...\n")
    results = orchestrator.run_full_experiment(
        num_samples=args.num_samples,
        architectures=architectures,
        models=models
    )

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    summary = orchestrator.get_summary_stats()

    for arch, arch_results in summary.items():
        print(f"\n{arch.upper().replace('_', ' ')}:")
        for model, stats in arch_results.items():
            print(f"  {model}:")
            print(f"    Total evaluations: {stats['total_samples']}")
            print(f"    Correct: {stats['correct']}")
            print(f"    Overall Accuracy: {stats['accuracy']:.2%}")

            # Display detailed metrics if available
            if 'FA' in stats:
                print(f"    ─────────────────────────────")
                print(f"    FA (First Accuracy):  {stats['FA']:.2%}")
                print(f"    SA (Second Accuracy): {stats['SA']:.2%}")
                print(f"    AA (Average Accuracy): {stats['AA']:.2%}")
                print(f"    CA (Consistent Accuracy): {stats['CA']:.2%}")
                print(f"    Position Bias: {stats['position_bias']:.2%}")
                print(f"    Paired samples: {stats['total_paired']}")

    print("\n" + "=" * 80)
    print("Results saved to:")
    print(f"  - results/raw/")
    print(f"  - results/processed/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
