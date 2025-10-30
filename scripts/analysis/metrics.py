"""
Metrics Calculation Module

Implements WiserUI-Bench metrics:
- CA (Consistent Accuracy): Correct in BOTH position orders
- FA (First Accuracy): Correct when winner shown first
- SA (Second Accuracy): Correct when winner shown second
- AA (Average Accuracy): (FA + SA) / 2
- Position Bias: |FA - SA| / 2

Reference: https://arxiv.org/html/2505.05026v3
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class MetricsCalculator:
    """
    Calculate WiserUI-Bench metrics from experiment results.

    Metrics:
    - FA: First Accuracy (correct when winner shown first)
    - SA: Second Accuracy (correct when winner shown second)
    - AA: Average Accuracy = (FA + SA) / 2
    - CA: Consistent Accuracy (correct in BOTH orders)
    - Position Bias: |FA - SA| / 2
    """

    def __init__(self, results: Dict[str, Any]):
        """
        Initialize metrics calculator.

        Args:
            results: Experiment results dictionary
        """
        self.results = results

    @staticmethod
    def load_results(results_path: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)

    def calculate_accuracy_by_position(
        self,
        architecture: str,
        model: str
    ) -> Dict[str, float]:
        """
        Calculate FA and SA for a model.

        Args:
            architecture: 'spot_and_judge' or 'baseline_e2e'
            model: Model name

        Returns:
            Dict with FA, SA, AA, position_bias
        """
        results = self.results[architecture][model]

        # Group by sample_id
        by_sample = defaultdict(dict)
        for result in results:
            sample_id = result['sample_id']
            position = result['position_order']
            by_sample[sample_id][position] = result['correct']

        # Calculate FA and SA
        fa_count = 0  # Correct when win shown first
        sa_count = 0  # Correct when win shown second
        total_first = 0
        total_second = 0

        for sample_id, positions in by_sample.items():
            if 'win_first' in positions:
                if positions['win_first']:
                    fa_count += 1
                total_first += 1

            if 'win_second' in positions:
                if positions['win_second']:
                    sa_count += 1
                total_second += 1

        fa = fa_count / total_first if total_first > 0 else 0
        sa = sa_count / total_second if total_second > 0 else 0
        aa = (fa + sa) / 2
        position_bias = abs(fa - sa) / 2

        return {
            'FA': fa,
            'SA': sa,
            'AA': aa,
            'position_bias': position_bias,
            'fa_count': fa_count,
            'sa_count': sa_count,
            'total_first': total_first,
            'total_second': total_second
        }

    def calculate_consistent_accuracy(
        self,
        architecture: str,
        model: str
    ) -> Dict[str, float]:
        """
        Calculate CA (Consistent Accuracy).

        CA = Correct in BOTH position orders (order-invariant metric)

        Args:
            architecture: 'spot_and_judge' or 'baseline_e2e'
            model: Model name

        Returns:
            Dict with CA and related stats
        """
        results = self.results[architecture][model]

        # Group by sample_id
        by_sample = defaultdict(dict)
        for result in results:
            sample_id = result['sample_id']
            position = result['position_order']
            by_sample[sample_id][position] = result['correct']

        # Calculate CA: Correct in BOTH orders
        consistent_correct = 0
        inconsistent = 0
        consistent_wrong = 0
        total_paired = 0

        for sample_id, positions in by_sample.items():
            # Only count samples with both positions tested
            if 'win_first' in positions and 'win_second' in positions:
                first_correct = positions['win_first']
                second_correct = positions['win_second']

                if first_correct and second_correct:
                    consistent_correct += 1
                elif first_correct != second_correct:
                    inconsistent += 1
                else:  # Both wrong
                    consistent_wrong += 1

                total_paired += 1

        ca = consistent_correct / total_paired if total_paired > 0 else 0
        consistency_rate = (consistent_correct + consistent_wrong) / total_paired if total_paired > 0 else 0

        return {
            'CA': ca,
            'consistent_correct': consistent_correct,
            'inconsistent': inconsistent,
            'consistent_wrong': consistent_wrong,
            'total_paired': total_paired,
            'consistency_rate': consistency_rate
        }

    def calculate_all_metrics(
        self,
        architecture: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for a model.

        Returns:
            Dict with FA, SA, AA, CA, position_bias
        """
        position_metrics = self.calculate_accuracy_by_position(architecture, model)
        consistency_metrics = self.calculate_consistent_accuracy(architecture, model)

        return {
            **position_metrics,
            **consistency_metrics
        }

    def compare_architectures(
        self,
        model: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare Spot & Judge vs Baseline for a specific model.

        Args:
            model: Model name (must exist in both architectures)

        Returns:
            Dict with metrics for both architectures
        """
        comparison = {}

        if model in self.results.get('spot_and_judge', {}):
            comparison['spot_and_judge'] = self.calculate_all_metrics('spot_and_judge', model)

        if model in self.results.get('baseline_e2e', {}):
            comparison['baseline_e2e'] = self.calculate_all_metrics('baseline_e2e', model)

        return comparison

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of all metrics.

        Returns:
            DataFrame with all models and architectures
        """
        rows = []

        for arch in ['spot_and_judge', 'baseline_e2e']:
            if arch not in self.results:
                continue

            for model in self.results[arch].keys():
                metrics = self.calculate_all_metrics(arch, model)

                rows.append({
                    'Architecture': arch.replace('_', ' ').title(),
                    'Model': model,
                    'FA': f"{metrics['FA']:.2%}",
                    'SA': f"{metrics['SA']:.2%}",
                    'AA': f"{metrics['AA']:.2%}",
                    'CA': f"{metrics['CA']:.2%}",
                    'Position Bias': f"{metrics['position_bias']:.2%}",
                    'Samples': metrics['total_paired']
                })

        return pd.DataFrame(rows)

    def calculate_improvement(
        self,
        model: str
    ) -> Dict[str, float]:
        """
        Calculate improvement of Spot & Judge over Baseline.

        Args:
            model: Model name

        Returns:
            Dict with absolute and relative improvements
        """
        comparison = self.compare_architectures(model)

        if 'spot_and_judge' not in comparison or 'baseline_e2e' not in comparison:
            return {}

        sj = comparison['spot_and_judge']
        bl = comparison['baseline_e2e']

        improvements = {}
        for metric in ['FA', 'SA', 'AA', 'CA']:
            abs_improvement = sj[metric] - bl[metric]
            rel_improvement = (abs_improvement / bl[metric] * 100) if bl[metric] > 0 else 0

            improvements[metric] = {
                'absolute': abs_improvement,
                'relative_pct': rel_improvement,
                'baseline': bl[metric],
                'spot_and_judge': sj[metric]
            }

        # Position bias reduction
        bias_reduction = bl['position_bias'] - sj['position_bias']
        improvements['position_bias_reduction'] = {
            'absolute': bias_reduction,
            'baseline': bl['position_bias'],
            'spot_and_judge': sj['position_bias']
        }

        return improvements

    def export_to_csv(self, output_path: str):
        """Export metrics summary to CSV."""
        df = self.generate_summary_table()
        df.to_csv(output_path, index=False)
        print(f"Metrics exported to: {output_path}")

    def export_detailed_json(self, output_path: str):
        """Export detailed metrics to JSON."""
        detailed = {}

        for arch in ['spot_and_judge', 'baseline_e2e']:
            if arch not in self.results:
                continue

            detailed[arch] = {}
            for model in self.results[arch].keys():
                detailed[arch][model] = self.calculate_all_metrics(arch, model)

        with open(output_path, 'w') as f:
            json.dump(detailed, f, indent=2)

        print(f"Detailed metrics exported to: {output_path}")


def analyze_results(results_path: str, output_dir: str = "results/processed"):
    """
    Convenience function to analyze results and export metrics.

    Args:
        results_path: Path to experiment results JSON
        output_dir: Directory to save analysis outputs
    """
    # Load results
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Calculate metrics
    print("\nCalculating metrics...")
    calculator = MetricsCalculator(results)

    # Generate summary table
    summary_df = calculator.generate_summary_table()
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Export results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = Path(results_path).stem.split('_')[-1]

    # Export CSV
    csv_path = output_path / f"metrics_summary_{timestamp}.csv"
    calculator.export_to_csv(str(csv_path))

    # Export detailed JSON
    json_path = output_path / f"metrics_detailed_{timestamp}.json"
    calculator.export_detailed_json(str(json_path))

    # Calculate improvements (if both architectures present)
    for model in results.get('spot_and_judge', {}).keys():
        if model in results.get('baseline_e2e', {}):
            print(f"\n{model.upper()} - Spot & Judge vs Baseline:")
            improvements = calculator.calculate_improvement(model)
            for metric, values in improvements.items():
                if metric == 'position_bias_reduction':
                    print(f"  Position Bias Reduction: {values['absolute']:+.2%}")
                else:
                    print(f"  {metric}: {values['absolute']:+.2%} ({values['relative_pct']:+.1f}%)")

    return calculator


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        print("Usage: python metrics.py <results_json_path>")
