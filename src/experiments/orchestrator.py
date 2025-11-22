"""
Experiment Orchestrator

Coordinates full experimental runs comparing Spot & Judge architecture
against baseline E2E evaluation on WiserUI-Bench dataset.

Implements WiserUI-Bench protocol:
- Position permutation testing (win-first, win-second)
- Multiple independent runs
- Metrics: CA, FA, SA, AA
- Checkpointing for resumption
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import yaml
from tqdm import tqdm

from ..data.dataset_loader import WiserUIBenchLoader
from ..models.glm4v_spotter import GLM4VSpotter
from ..models.llm_judger import LLMJudger
from ..models.baseline_e2e import BaselineE2EEvaluator
from ..utils.logger import ExperimentLogger

# Import metrics calculator (use relative import to avoid circular dependency)
import sys
from pathlib import Path
metrics_path = Path(__file__).parent.parent.parent / 'scripts' / 'analysis'
if str(metrics_path) not in sys.path:
    sys.path.insert(0, str(metrics_path))
from metrics import MetricsCalculator


class ExperimentOrchestrator:
    """
    Orchestrates full experimental runs.

    Manages:
    - Dataset iteration
    - Position permutation testing
    - Multiple architecture comparison
    - Result collection and checkpointing
    """

    def __init__(
        self,
        config_path: str = "config/experiment_config.yaml",
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Initialize experiment orchestrator.

        Args:
            config_path: Path to experiment configuration
            resume_from_checkpoint: Optional checkpoint file to resume from
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize logger
        self.logger = ExperimentLogger(config_path)

        # Initialize dataset loader
        self.dataset_loader = WiserUIBenchLoader(config_path=config_path)
        self.dataset_loader.load(split='test')

        # Initialize models (lazy - only when needed)
        self.spotter = None
        self.judgers = {}
        self.baseline_evaluators = {}

        # Experiment tracking
        self.results = {
            'spot_and_judge': {},
            'baseline_e2e': {}
        }
        self.completed_samples = set()

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

        # Spotter cache directory
        self.spotter_cache_dir = Path(self.config.get('results', {}).get('output_dir', './results')) / 'spotter_cache'
        self.spotter_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_spotter_cache_path(self, sample_id: str) -> Path:
        """Get path for spotter cache file."""
        return self.spotter_cache_dir / f"{sample_id}.json"

    def _load_spotter_cache(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Load spotter result from cache."""
        cache_path = self._get_spotter_cache_path(sample_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load spotter cache for {sample_id}: {e}")
        return None

    def _save_spotter_cache(self, sample_id: str, data: Dict[str, Any]):
        """Save spotter result to cache."""
        cache_path = self._get_spotter_cache_path(sample_id)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save spotter cache for {sample_id}: {e}")

    def _init_spotter(self):
        """Lazy initialize Spotter."""
        if self.spotter is None:
            self.logger.info("Initializing GLM-4.5V Spotter...")
            self.spotter = GLM4VSpotter()
            self.logger.info("Spotter initialized")

    def _init_judger(self, model_name: str):
        """Lazy initialize Judger for specific model."""
        if model_name not in self.judgers:
            self.logger.info(f"Initializing {model_name} Judger...")
            self.judgers[model_name] = LLMJudger(model_name=model_name)
            self.logger.info(f"Judger {model_name} initialized")

    def _init_baseline(self, model_name: str):
        """Lazy initialize Baseline evaluator."""
        if model_name not in self.baseline_evaluators:
            self.logger.info(f"Initializing {model_name} Baseline...")
            self.baseline_evaluators[model_name] = BaselineE2EEvaluator(model_name=model_name)
            self.logger.info(f"Baseline {model_name} initialized")

    def run_spot_and_judge(
        self,
        sample: Dict[str, Any],
        judger_model: str,
        position_order: str
    ) -> Dict[str, Any]:
        """
        Run Spot & Judge architecture on one sample.

        Args:
            sample: Dataset sample
            judger_model: Which model to use for Judger
            position_order: 'win_first' or 'win_second'

        Returns:
            Result dictionary with judgment and metadata
        """
        self._init_spotter()
        self._init_judger(judger_model)

        if position_order == 'win_first':
            spotter_baseline = sample['image_win_path']
            spotter_variant = sample['image_lose_path']
            judger_baseline = sample['image_win_path']
            judger_variant = sample['image_lose_path']
        else:
            spotter_baseline = sample['image_lose_path']
            spotter_variant = sample['image_win_path']
            judger_baseline = sample['image_lose_path']
            judger_variant = sample['image_win_path']

        # Stage 1: Spotter (with Caching & Combination)
        # We always cache the "Win-First" perspective (Win=Image1, Lose=Image2)
        
        # Check cache first
        cached_spotter_data = self._load_spotter_cache(sample['id'])
        
        if cached_spotter_data:
            self.logger.info(f"Using cached spotter result for {sample['id']}")
            spotter_output = cached_spotter_data['output']
            spotter_metadata = cached_spotter_data['metadata']
            spotter_time = 0  # Cached
        else:
            # Not cached: Run Spotter TWICE and combine
            self.logger.info(f"Running spotter for {sample['id']} (Combined Mode)...")
            start_time = time.time()
            
            # Run 1: Win -> Lose
            out_wl, meta_wl = self.spotter.spot_differences(
                image_a_path=sample['image_win_path'],
                image_b_path=sample['image_lose_path']
            )
            
            # Run 2: Lose -> Win
            out_lw, meta_lw = self.spotter.spot_differences(
                image_a_path=sample['image_lose_path'],
                image_b_path=sample['image_win_path']
            )
            
            # Swap Run 2 results to match Win-First perspective
            out_lw_swapped = self.spotter.swap_differences(out_lw)
            
            # Combine results
            combined_differences = out_wl.get('differences', []) + out_lw_swapped.get('differences', [])
            spotter_output = {"differences": combined_differences}
            
            # Merge metadata (sum tokens, max latency)
            spotter_metadata = {
                "latency_seconds": meta_wl.get('latency_seconds', 0) + meta_lw.get('latency_seconds', 0),
                "input_tokens": meta_wl.get('input_tokens', 0) + meta_lw.get('input_tokens', 0),
                "output_tokens": meta_wl.get('output_tokens', 0) + meta_lw.get('output_tokens', 0),
                "total_tokens": meta_wl.get('total_tokens', 0) + meta_lw.get('total_tokens', 0),
                "model": meta_wl.get('model', 'unknown'),
                "combined": True
            }
            spotter_time = time.time() - start_time
            
            # Save to cache
            self._save_spotter_cache(sample['id'], {
                'output': spotter_output,
                'metadata': spotter_metadata
            })

        # If current order is Win-Second (Lose -> Win), we need to swap the cached Win-First result
        if position_order == 'win_second':
            spotter_output = self.spotter.swap_differences(spotter_output)

        # Log Spotter Usage (even if cached, we track it for the run)
        self.logger.log_api_call(
            model='glm-4.5v',
            prompt_type='spotter',
            input_tokens=spotter_metadata['input_tokens'],
            output_tokens=spotter_metadata['output_tokens'],
            latency_seconds=spotter_time,
            prompt="Combined Spotter Run (Cached/Fresh)",
            response=json.dumps(spotter_output, indent=2),
            metadata={'sample_id': sample['id'], 'position_order': position_order, 'cached': (spotter_time == 0)}
        )

        # Stage 2: Judger
        start_time = time.time()
        judgment, reasoning, judger_metadata = self.judgers[judger_model].judge(
            spotter_output=spotter_output,
            baseline_image_path=judger_baseline
        )
        judger_time = time.time() - start_time

        # Log Judger API call
        self.logger.log_api_call(
            model=judger_model,
            prompt_type='judger',
            input_tokens=judger_metadata['input_tokens'],
            output_tokens=judger_metadata['output_tokens'],
            latency_seconds=judger_time,
            prompt=json.dumps(judger_metadata.get('prompt', {}), indent=2),
            response=json.dumps(reasoning, indent=2),
            metadata={
                'sample_id': sample['id'],
                'position_order': position_order,
                'judgment': judgment,
                'spotter_output': spotter_output
            }
        )

        # Determine correctness
        if position_order == 'win_first':
            correct = (judgment == 'first')
        else:  # win_second
            correct = (judgment == 'second')

        return {
            'sample_id': sample['id'],
            'position_order': position_order,
            'judgment': judgment,
            'correct': correct,
            'spotter_output': spotter_output,
            'judger_reasoning': reasoning,
            'total_time': spotter_time + judger_time,
            'spotter_tokens': spotter_metadata['total_tokens'],
            'judger_tokens': judger_metadata['total_tokens']
        }

    def run_baseline(
        self,
        sample: Dict[str, Any],
        model_name: str,
        position_order: str
    ) -> Dict[str, Any]:
        """
        Run Baseline E2E evaluation on one sample.

        Args:
            sample: Dataset sample
            model_name: Which model to use
            position_order: 'win_first' or 'win_second'

        Returns:
            Result dictionary with judgment and metadata
        """
        self._init_baseline(model_name)

        # Determine image order (baseline receives images in the right order)
        if position_order == 'win_first':
            first_image = sample['image_win_path']
            second_image = sample['image_lose_path']
        else:  # win_second
            first_image = sample['image_lose_path']
            second_image = sample['image_win_path']

        # Run baseline evaluation
        start_time = time.time()
        judgment, metadata = self.baseline_evaluators[model_name].evaluate(
            image_a_path=first_image,
            image_b_path=second_image
        )
        eval_time = time.time() - start_time

        # Log API call
        self.logger.log_api_call(
            model=model_name,
            prompt_type='baseline',
            input_tokens=metadata.get('input_tokens', 0),
            output_tokens=metadata.get('output_tokens', 0),
            latency_seconds=eval_time,
            prompt="WiserUI-Bench baseline prompt (see prompt_templates.py)",
            response=metadata.get('response', ''),
            metadata={
                'sample_id': sample['id'],
                'position_order': position_order,
                'judgment': judgment
            }
        )

        # Determine correctness
        if position_order == 'win_first':
            correct = (judgment == 'first')
        else:  # win_second
            correct = (judgment == 'second')

        return {
            'sample_id': sample['id'],
            'position_order': position_order,
            'judgment': judgment,
            'correct': correct,
            'response': metadata.get('response', ''),
            'total_time': eval_time,
            'total_tokens': metadata.get('total_tokens', 0)
        }

    def run_full_experiment(
        self,
        num_samples: Optional[int] = None,
        architectures: Optional[List[str]] = None,
        models: Optional[Dict[str, List[str]]] = None
    ):
        """
        Run complete experiment on dataset.

        Args:
            num_samples: Limit number of samples (None = all)
            architectures: Which architectures to test (['spot_and_judge', 'baseline_e2e'])
            models: Which models to use for each architecture
        """
        # Get samples
        all_samples = self.dataset_loader.get_all_samples()
        if num_samples:
            samples = all_samples[:num_samples]
        else:
            samples = all_samples

        # Default architectures
        if architectures is None:
            architectures = ['spot_and_judge', 'baseline_e2e']

        # Default models
        if models is None:
            models = {
                'spot_and_judge': ['gpt-4o', 'claude-3.5-sonnet'],
                'baseline_e2e': ['gpt-4o', 'claude-3.5-sonnet']
            }

        # Position orders
        position_orders = ['win_first', 'win_second']

        # Calculate total iterations
        total_iterations = 0
        if 'spot_and_judge' in architectures:
            total_iterations += len(samples) * len(models['spot_and_judge']) * len(position_orders)
        if 'baseline_e2e' in architectures:
            total_iterations += len(samples) * len(models['baseline_e2e']) * len(position_orders)

        self.logger.log_experiment_start(
            architecture=f"Spot&Judge + Baseline",
            config={
                'num_samples': len(samples),
                'architectures': architectures,
                'models': models,
                'total_iterations': total_iterations
            }
        )

        # Progress bar
        pbar = tqdm(total=total_iterations, desc="Running experiment")

        try:
            # Run Spot & Judge
            if 'spot_and_judge' in architectures:
                for model in models['spot_and_judge']:
                    # Initialize list if not present, but PRESERVE if resuming
                    if model not in self.results['spot_and_judge']:
                        self.results['spot_and_judge'][model] = []

                    for sample in samples:
                        for position in position_orders:
                            # Check if already processed
                            already_processed = False
                            for r in self.results['spot_and_judge'][model]:
                                if r['sample_id'] == sample['id'] and r['position_order'] == position:
                                    already_processed = True
                                    break
                            
                            if already_processed:
                                # self.logger.info(f"Skipping {sample['id']} ({model}, {position}) - Already processed")
                                pbar.update(1)
                                continue

                            try:
                                result = self.run_spot_and_judge(sample, model, position)
                                self.results['spot_and_judge'][model].append(result)
                                pbar.update(1)

                                # Checkpoint every 10 samples
                                if len(self.results['spot_and_judge'][model]) % 10 == 0:
                                    self._save_checkpoint()

                            except Exception as e:
                                self.logger.error(f"Error on {sample['id']} ({model}, {position}): {str(e)}")
                                pbar.update(1)
                                continue

            # Run Baseline
            if 'baseline_e2e' in architectures:
                for model in models['baseline_e2e']:
                    # Initialize list if not present, but PRESERVE if resuming
                    if model not in self.results['baseline_e2e']:
                        self.results['baseline_e2e'][model] = []

                    for sample in samples:
                        for position in position_orders:
                            # Check if already processed
                            already_processed = False
                            for r in self.results['baseline_e2e'][model]:
                                if r['sample_id'] == sample['id'] and r['position_order'] == position:
                                    already_processed = True
                                    break
                            
                            if already_processed:
                                # self.logger.info(f"Skipping {sample['id']} ({model}, {position}) - Already processed")
                                pbar.update(1)
                                continue

                            try:
                                result = self.run_baseline(sample, model, position)
                                self.results['baseline_e2e'][model].append(result)
                                pbar.update(1)

                                # Checkpoint every 10 samples
                                if len(self.results['baseline_e2e'][model]) % 10 == 0:
                                    self._save_checkpoint()

                            except Exception as e:
                                self.logger.error(f"Error on {sample['id']} ({model}, {position}): {str(e)}")
                                pbar.update(1)
                                continue

        finally:
            pbar.close()

        # Save final results
        self._save_results()

        # Log completion
        self.logger.log_experiment_end(
            results_summary=self.get_summary_stats()
        )

        return self.results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of experiment results including FA, SA, AA, CA metrics."""
        summary = {}

        # First calculate basic stats
        for arch in ['spot_and_judge', 'baseline_e2e']:
            summary[arch] = {}
            for model, results in self.results[arch].items():
                if not results:
                    continue

                total = len(results)
                correct = sum(1 for r in results if r['correct'])
                accuracy = correct / total if total > 0 else 0

                summary[arch][model] = {
                    'total_samples': total,
                    'correct': correct,
                    'accuracy': accuracy
                }

        # Calculate detailed metrics using MetricsCalculator if we have position data
        try:
            calculator = MetricsCalculator(self.results)

            for arch in ['spot_and_judge', 'baseline_e2e']:
                if arch not in self.results:
                    continue

                for model in self.results[arch].keys():
                    if not self.results[arch][model]:
                        continue

                    metrics = calculator.calculate_all_metrics(arch, model)

                    # Add detailed metrics to summary
                    summary[arch][model].update({
                        'FA': metrics['FA'],
                        'SA': metrics['SA'],
                        'AA': metrics['AA'],
                        'CA': metrics['CA'],
                        'position_bias': metrics['position_bias'],
                        'total_paired': metrics['total_paired']
                    })
        except Exception as e:
            # If metrics calculation fails, just use basic stats
            self.logger.warning(f"Could not calculate detailed metrics: {str(e)}")

        return summary

    def _save_checkpoint(self):
        """Save checkpoint for resumption."""
        checkpoint_data = {
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.log_checkpoint(checkpoint_data)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        self.results = checkpoint['results']
        self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_results(self):
        """Save final results to disk."""
        results_dir = Path(self.config.get('results', {}).get('output_dir', './results'))
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results as JSON
        results_file = results_dir / 'raw' / f"experiment_results_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Results saved to: {results_file}")

        # Also save summary
        summary_file = results_dir / 'processed' / f"summary_{timestamp}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(self.get_summary_stats(), f, indent=2)

        self.logger.info(f"Summary saved to: {summary_file}")
