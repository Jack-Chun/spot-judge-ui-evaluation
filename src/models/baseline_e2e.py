"""
Baseline End-to-End Evaluation Module

This module implements the standard MLLM evaluation approach from WiserUI-Bench.
Models directly view both UI images and make a judgment.

Purpose: Baseline comparison to measure Spot & Judge improvement.
Configuration: EXACTLY matches WiserUI-Bench experimental setup (temp=0.2, etc.)
"""

import os
import time
import json
import base64
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml

from openai import OpenAI
from anthropic import Anthropic

from ..utils.prompt_templates import (
    format_baseline_prompt,
    parse_baseline_response
)


class BaselineE2EEvaluator:
    """
    Baseline E2E evaluator replicating WiserUI-Bench methodology.

    Direct MLLM evaluation: Model sees both images and makes judgment.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        config_path: str = "config/models_config.yaml"
    ):
        """
        Initialize baseline evaluator.

        Args:
            model_name: Model to use ('gpt-4o' or 'claude-3.5-sonnet')
            api_key: API key
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.baseline_config = config['baseline']

        self.model_name = model_name.lower()

        # Temperature MUST be 0.2 (WiserUI-Bench baseline)
        # Hardcode to ensure exact replication
        self.temperature = 0.2
        assert self.temperature == 0.2, "WiserUI-Bench baseline requires temperature=0.2"
        self.max_tokens = self.baseline_config.get('max_tokens', 2000)

        # Initialize model based on type
        if 'gpt' in self.model_name:
            self.llm = self._init_openai(api_key, config)
            self.provider = 'openai'
            self.supports_vision = True
        elif 'claude' in self.model_name:
            self.llm = self._init_anthropic(api_key, config)
            self.provider = 'anthropic'
            self.supports_vision = True
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'gpt-4o' or 'claude-3.5-sonnet'.")

    def _init_openai(self, api_key: Optional[str], config: Dict) -> OpenAI:
        """Initialize OpenAI native client."""
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        model_config = config['baseline']['models']['gpt-4o']
        self.openai_model_name = model_config['model_name']

        return OpenAI(api_key=api_key)

    def _init_anthropic(self, api_key: Optional[str], config: Dict) -> Anthropic:
        """Initialize Anthropic native client."""
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        model_config = config['baseline']['models']['claude-3.5-sonnet']
        self.anthropic_model_name = model_config['model_name']

        return Anthropic(api_key=api_key)

    def evaluate(
        self,
        image_a_path: str,
        image_b_path: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluate which UI is more effective (baseline E2E approach).

        Orchestrator handles position permutation - this method evaluates
        images in the order provided (A shown first, B shown second).

        Args:
            image_a_path: Path to first UI image (shown first)
            image_b_path: Path to second UI image (shown second)

        Returns:
            Tuple of (judgment, metadata) where:
            - judgment: 'first' or 'second'
            - metadata: API call metadata
        """
        # Get prompts (WiserUI-Bench format)
        prompts = format_baseline_prompt(position="first_second")

        # Encode images (A is first, B is second)
        first_image_b64 = self._encode_image(image_a_path)
        second_image_b64 = self._encode_image(image_b_path)

        # Call model based on provider
        if self.provider == 'openai':
            judgment, metadata = self._call_openai_native(
                prompts, first_image_b64, second_image_b64
            )
        elif self.provider == 'anthropic':
            judgment, metadata = self._call_anthropic_native(
                prompts, first_image_b64, second_image_b64
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Add image paths to metadata
        metadata['image_a'] = image_a_path
        metadata['image_b'] = image_b_path

        return judgment, metadata

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 - EXACTLY matching WiserUI-Bench method.

        Reference: https://github.com/jeochris/wiserui-bench/blob/main/inference/VLM.py#L41-L45
        """
        from PIL import Image
        from io import BytesIO

        # Load image
        image = Image.open(image_path)

        # Convert to RGB (handles RGBA, P modes)
        image = image.convert("RGB")

        # Save as PNG to buffer (matching WiserUI-Bench)
        buffered = BytesIO()
        image.save(buffered, format="PNG")

        # Encode and return with data URI
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _call_openai_native(
        self,
        prompts: Dict[str, str],
        first_image_b64: str,
        second_image_b64: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Call OpenAI model via native API."""
        # Create messages for OpenAI Chat Completions API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts['prompt']},
                    {"type": "image_url", "image_url": {"url": first_image_b64}},
                    {"type": "image_url", "image_url": {"url": second_image_b64}}
                ]
            }
        ]

        start_time = time.time()

        try:
            response = self.llm.chat.completions.create(
                model=self.openai_model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            latency = time.time() - start_time

            content = response.choices[0].message.content

            # Get token usage from native response
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            metadata = {
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "model": self.model_name,
                "provider": self.provider,
                "response": content
            }

            # Parse judgment
            judgment = parse_baseline_response(content)

            return judgment, metadata

        except Exception as e:
            raise RuntimeError(f"Baseline evaluation failed: {str(e)}") from e

    def _call_anthropic_native(
        self,
        prompts: Dict[str, str],
        first_image_b64: str,
        second_image_b64: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Call Anthropic model via native API."""
        # Extract base64 data from data URI
        first_b64_data = first_image_b64.split(',')[1] if ',' in first_image_b64 else first_image_b64
        second_b64_data = second_image_b64.split(',')[1] if ',' in second_image_b64 else second_image_b64

        # Create messages with images (Anthropic format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts['prompt']},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": first_b64_data
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": second_b64_data
                        }
                    }
                ]
            }
        ]

        start_time = time.time()

        try:
            response = self.llm.messages.create(
                model=self.anthropic_model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages
            )
            latency = time.time() - start_time

            content = response.content[0].text

            # Get token usage from native response
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            metadata = {
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "model": self.model_name,
                "provider": self.provider,
                "response": content
            }

            # Parse judgment
            judgment = parse_baseline_response(content)

            return judgment, metadata

        except Exception as e:
            raise RuntimeError(f"Baseline evaluation failed: {str(e)}") from e

