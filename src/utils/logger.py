"""
Logging utilities for experiment tracking and cost management.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ExperimentLogger:
    """Logger for tracking experimental runs, API calls, and costs."""

    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """
        Initialize experiment logger.

        Args:
            config_path: Path to experiment configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Session info (must be set BEFORE logger setup)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up logging directory
        log_config = self.config.get('logging', {})
        self.log_dir = Path(log_config.get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session log file
        self.session_log_file = self.log_dir / f"session_{self.session_id}.jsonl"

        # Initialize Python logger
        self.logger = self._setup_logger(
            level=log_config.get('level', 'INFO'),
            format_str=log_config.get('log_format')
        )

        # Cost tracking
        self.cost_tracking_enabled = self.config.get('cost_management', {}).get('enable_tracking', True)
        self.total_cost = 0.0
        self.cost_by_model = {}

        # API call tracking
        self.api_calls = []

    def _setup_logger(self, level: str, format_str: Optional[str] = None) -> logging.Logger:
        """Set up Python logger with file and console handlers."""
        logger = logging.getLogger('SpotAndJudge')
        logger.setLevel(getattr(logging, level.upper()))

        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"experiment_{self.session_id}.log"
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        if format_str is None:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(format_str)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_api_call(
        self,
        model: str,
        prompt_type: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an API call with token usage and cost.

        Args:
            model: Model name (e.g., 'gpt-4o', 'glm-4.5v')
            prompt_type: Type of prompt ('spotter', 'judger', 'baseline')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_seconds: API call latency
            prompt: Optional prompt text
            response: Optional response text
            metadata: Optional additional metadata
        """
        # Calculate cost
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        # Update totals
        if self.cost_tracking_enabled:
            self.total_cost += cost
            self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost

        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "model": model,
            "prompt_type": prompt_type,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "cost_usd": round(cost, 6),
            "latency_seconds": round(latency_seconds, 3),
        }

        if prompt and self.config.get('logging', {}).get('log_prompts', True):
            log_entry["prompt"] = prompt

        if response and self.config.get('logging', {}).get('log_responses', True):
            log_entry["response"] = response

        if metadata:
            log_entry["metadata"] = metadata

        # Append to session log file
        with open(self.session_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Log to console if enabled
        if self.config.get('logging', {}).get('log_api_calls', True):
            self.logger.info(
                f"API call: {model} | {prompt_type} | "
                f"Tokens: {input_tokens}→{output_tokens} | "
                f"Cost: ${cost:.4f} | "
                f"Latency: {latency_seconds:.2f}s"
            )

        # Check cost limits
        self._check_cost_limits()

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call based on model pricing."""
        # Load pricing from config
        with open("config/models_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        pricing = config.get('pricing', {}).get(model, {})
        if not pricing:
            self.logger.warning(f"No pricing info for model: {model}")
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing.get('input', 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get('output', 0)

        return input_cost + output_cost

    def _check_cost_limits(self):
        """Check if cost limits have been reached."""
        cost_config = self.config.get('cost_management', {})

        if not cost_config.get('enable_tracking', True):
            return

        max_cost = cost_config.get('max_cost_per_run', 100.0)
        warn_threshold = cost_config.get('warn_threshold', 80.0)

        # Warning threshold
        if self.total_cost >= (max_cost * warn_threshold / 100):
            self.logger.warning(
                f"Cost warning: ${self.total_cost:.2f} / ${max_cost:.2f} "
                f"({self.total_cost/max_cost*100:.1f}%)"
            )

        # Hard limit
        if self.total_cost >= max_cost:
            error_msg = f"Cost limit reached: ${self.total_cost:.2f} / ${max_cost:.2f}"
            if cost_config.get('stop_on_limit', True):
                raise RuntimeError(error_msg)
            else:
                self.logger.error(error_msg)

    def log_experiment_start(self, architecture: str, config: Dict[str, Any]):
        """Log the start of an experiment run."""
        self.logger.info("=" * 80)
        self.logger.info(f"Starting experiment: {architecture}")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        self.logger.info("=" * 80)

    def log_experiment_end(self, results_summary: Dict[str, Any]):
        """Log the end of an experiment run."""
        self.logger.info("=" * 80)
        self.logger.info("Experiment completed")
        self.logger.info(f"Total cost: ${self.total_cost:.4f}")
        self.logger.info("Cost by model:")
        for model, cost in self.cost_by_model.items():
            self.logger.info(f"  {model}: ${cost:.4f}")
        self.logger.info(f"Results: {json.dumps(results_summary, indent=2)}")
        self.logger.info("=" * 80)

    def log_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Log a checkpoint for experiment resumption."""
        checkpoint_dir = Path(self.config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint_{self.session_id}.json"

        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "total_cost": self.total_cost,
            "cost_by_model": self.cost_by_model,
            **checkpoint_data
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.info(f"Checkpoint saved: {checkpoint_file}")

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "cost_by_model": {
                model: round(cost, 4)
                for model, cost in self.cost_by_model.items()
            },
            "session_id": self.session_id
        }

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
