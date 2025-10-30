"""
Run a pilot experiment on a small subset to test the full pipeline.

This runs the same experiment as run_experiment.py but on only 5 samples
to verify everything works before running the full 300-sample experiment.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.orchestrator import ExperimentOrchestrator


def main():
    print("=" * 80)
    print("PILOT EXPERIMENT (5 samples)")
    print("=" * 80)
    print("Testing both architectures with GPT-4o")
    print("Expected cost: ~$2-5")
    print("Expected time: ~10-15 minutes")
    print("=" * 80)
    print()

    # Initialize orchestrator
    print("Initializing experiment orchestrator...")
    orchestrator = ExperimentOrchestrator()

    # Run on 5 samples with just GPT-4o
    print("\nStarting pilot experiment...\n")
    results = orchestrator.run_full_experiment(
        num_samples=5,
        architectures=['spot_and_judge', 'baseline_e2e'],
        models={
            'spot_and_judge': ['gpt-4o'],
            'baseline_e2e': ['gpt-4o']
        }
    )

    # Print summary
    print("\n" + "=" * 80)
    print("PILOT EXPERIMENT COMPLETE")
    print("=" * 80)

    summary = orchestrator.get_summary_stats()

    for arch, arch_results in summary.items():
        print(f"\n{arch.upper().replace('_', ' ')}:")
        for model, stats in arch_results.items():
            print(f"  {model}:")
            print(f"    Samples tested: {stats['total_samples'] // 2} (× 2 positions)")
            print(f"    Total evaluations: {stats['total_samples']}")
            print(f"    Correct: {stats['correct']}")
            print(f"    Accuracy: {stats['accuracy']:.2%}")

    print("\n" + "=" * 80)
    print("✓ Pilot successful! You can now run the full experiment with:")
    print("  python scripts/run_experiment.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
