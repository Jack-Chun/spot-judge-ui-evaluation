"""
Visualization Module

Creates publication-ready charts and figures for the paper:
- CA comparison bar charts
- Position bias heatmaps
- Statistical significance visualizations
- Model improvement trends
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .metrics import MetricsCalculator

# Set publication-ready style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

# Custom colors for Spot & Judge vs Baseline
COLORS = {
    'spot_and_judge': '#2E86AB',  # Blue
    'baseline_e2e': '#A23B72',     # Purple
    'improvement': '#06A77D'        # Green
}


class Visualizer:
    """Create visualizations for experimental results."""

    def __init__(
        self,
        metrics_calculator: MetricsCalculator,
        output_dir: str = "results/visualizations",
        dpi: int = 300
    ):
        """
        Initialize visualizer.

        Args:
            metrics_calculator: MetricsCalculator with results
            output_dir: Directory to save figures
            dpi: Resolution for saved figures
        """
        self.calculator = metrics_calculator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_ca_comparison(
        self,
        models: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Consistent Accuracy (CA) comparison bar chart.

        Primary metric showing Spot & Judge vs Baseline.

        Args:
            models: List of models to plot (None = all)
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        # Collect data
        data = []

        if models is None:
            # Get all models present in both architectures
            sj_models = set(self.calculator.results.get('spot_and_judge', {}).keys())
            bl_models = set(self.calculator.results.get('baseline_e2e', {}).keys())
            models = list(sj_models & bl_models)

        for model in models:
            # Spot & Judge
            if model in self.calculator.results.get('spot_and_judge', {}):
                sj_metrics = self.calculator.calculate_all_metrics('spot_and_judge', model)
                data.append({
                    'Model': model.upper(),
                    'Architecture': 'Spot & Judge',
                    'CA': sj_metrics['CA'] * 100
                })

            # Baseline
            if model in self.calculator.results.get('baseline_e2e', {}):
                bl_metrics = self.calculator.calculate_all_metrics('baseline_e2e', model)
                data.append({
                    'Model': model.upper(),
                    'Architecture': 'Baseline E2E',
                    'CA': bl_metrics['CA'] * 100
                })

        df = pd.DataFrame(data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot grouped bar chart
        x = np.arange(len(models))
        width = 0.35

        sj_data = df[df['Architecture'] == 'Spot & Judge']['CA'].values
        bl_data = df[df['Architecture'] == 'Baseline E2E']['CA'].values

        bars1 = ax.bar(x - width/2, sj_data, width, label='Spot & Judge',
                      color=COLORS['spot_and_judge'], alpha=0.8)
        bars2 = ax.bar(x + width/2, bl_data, width, label='Baseline E2E',
                      color=COLORS['baseline_e2e'], alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10)

        # Styling
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Consistent Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Consistent Accuracy (CA) Comparison\nSpot & Judge vs Baseline E2E',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add baseline reference line (25% - random chance for CA)
        ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Random (25%)')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / "ca_comparison.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def plot_all_metrics_comparison(
        self,
        model: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot all metrics (FA, SA, AA, CA) for one model.

        Args:
            model: Model name
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        # Get metrics
        comparison = self.calculator.compare_architectures(model)

        if not comparison:
            raise ValueError(f"Model {model} not found in results")

        sj = comparison.get('spot_and_judge', {})
        bl = comparison.get('baseline_e2e', {})

        # Prepare data
        metrics = ['FA', 'SA', 'AA', 'CA']
        sj_values = [sj.get(m, 0) * 100 for m in metrics]
        bl_values = [bl.get(m, 0) * 100 for m in metrics]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, sj_values, width, label='Spot & Judge',
                      color=COLORS['spot_and_judge'], alpha=0.8)
        bars2 = ax.bar(x + width/2, bl_values, width, label='Baseline E2E',
                      color=COLORS['baseline_e2e'], alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        # Styling
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'All Metrics Comparison - {model.upper()}\nSpot & Judge vs Baseline E2E',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"all_metrics_{model.lower().replace('-', '_')}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def plot_position_bias_heatmap(
        self,
        models: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot position bias heatmap.

        Shows FA vs SA for each model/architecture.

        Args:
            models: List of models to include
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        # Collect data
        rows = []

        if models is None:
            sj_models = set(self.calculator.results.get('spot_and_judge', {}).keys())
            bl_models = set(self.calculator.results.get('baseline_e2e', {}).keys())
            models = list(sj_models | bl_models)

        for model in models:
            for arch in ['spot_and_judge', 'baseline_e2e']:
                if model in self.calculator.results.get(arch, {}):
                    metrics = self.calculator.calculate_all_metrics(arch, model)
                    rows.append({
                        'Model': model.upper(),
                        'Architecture': 'S&J' if arch == 'spot_and_judge' else 'Baseline',
                        'FA': metrics['FA'] * 100,
                        'SA': metrics['SA'] * 100,
                        'Bias': metrics['position_bias'] * 100
                    })

        df = pd.DataFrame(rows)

        # Create pivot table for heatmap
        pivot = df.pivot_table(
            index=['Architecture', 'Model'],
            values='Bias'
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',  # Red = high bias, Green = low bias
            cbar_kws={'label': 'Position Bias (%)'},
            ax=ax,
            vmin=0,
            vmax=25,  # Scale to reasonable max
            linewidths=0.5
        )

        ax.set_title('Position Bias by Model and Architecture',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('', fontsize=12)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / "position_bias_heatmap.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def plot_improvement_bars(
        self,
        model: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot improvement bars showing Spot & Judge gains over Baseline.

        Args:
            model: Model name
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        improvements = self.calculator.calculate_improvement(model)

        if not improvements:
            raise ValueError(f"Cannot calculate improvements for {model}")

        # Prepare data
        metrics = ['FA', 'SA', 'AA', 'CA']
        abs_improvements = [improvements[m]['absolute'] * 100 for m in metrics]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(metrics, abs_improvements, color=COLORS['improvement'], alpha=0.8)

        # Color bars based on positive/negative
        for bar, val in zip(bars, abs_improvements):
            if val < 0:
                bar.set_color('#E63946')  # Red for negative

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, fontweight='bold')

        # Styling
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Spot & Judge Improvement Over Baseline - {model.upper()}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"improvement_{model.lower().replace('-', '_')}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def create_all_visualizations(self, models: Optional[List[str]] = None):
        """
        Generate all visualizations for the paper.

        Args:
            models: List of models to visualize (None = all)
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # 1. CA Comparison (main result)
        print("\n1. Creating CA comparison chart...")
        self.plot_ca_comparison(models=models)

        # 2. Position bias heatmap
        print("\n2. Creating position bias heatmap...")
        self.plot_position_bias_heatmap(models=models)

        # For each model:
        if models is None:
            sj_models = set(self.calculator.results.get('spot_and_judge', {}).keys())
            bl_models = set(self.calculator.results.get('baseline_e2e', {}).keys())
            models = list(sj_models & bl_models)

        for model in models:
            print(f"\n3. Creating detailed charts for {model}...")

            # All metrics comparison
            self.plot_all_metrics_comparison(model)

            # Improvement bars
            try:
                self.plot_improvement_bars(model)
            except ValueError:
                print(f"   Skipping improvement chart (insufficient data)")

        print("\n" + "="*80)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print("="*80)


def visualize_results(results_path: str, output_dir: str = "results/visualizations"):
    """
    Convenience function to create all visualizations from results file.

    Args:
        results_path: Path to experiment results JSON
        output_dir: Directory to save visualizations
    """
    # Load results and create calculator
    with open(results_path, 'r') as f:
        results = json.load(f)

    calculator = MetricsCalculator(results)

    # Create visualizer
    visualizer = Visualizer(calculator, output_dir=output_dir)

    # Generate all visualizations
    visualizer.create_all_visualizations()

    return visualizer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_results(sys.argv[1])
    else:
        print("Usage: python visualizations.py <results_json_path>")
