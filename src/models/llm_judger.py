"""
LLM Judger Module - Stage 2 of Spot & Judge Architecture

This module implements the "Judger" - a reasoning component that evaluates
UI effectiveness based on visual differences provided by the Spotter.

Key features:
- No direct image access (prevents position bias)
- UX principle-based reasoning
- Multiple LLM support (GPT-4o, Claude 3.5 Sonnet)
- Structured reasoning output
"""

import os
import time
import json
import base64
from typing import Dict, Any, Optional, Tuple
import yaml

from openai import OpenAI
from anthropic import Anthropic

from ..utils.prompt_templates import format_judger_prompt, parse_judger_response


class LLMJudger:
    """
    Judger using LLMs (GPT-4o, Claude) for UX-based reasoning.

    This is Stage 2 of the Spot & Judge architecture:
    Input: JSON describing visual differences (from Spotter)
    Output: Judgment of which variant is more effective + reasoning
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        config_path: str = "config/models_config.yaml"
    ):
        """
        Initialize LLM Judger.

        Args:
            model_name: Model to use ('gpt-4o' or 'claude-3.5-sonnet')
            api_key: API key (or use OPENAI_API_KEY / ANTHROPIC_API_KEY env vars)
            config_path: Path to model configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.judger_config = config['judger']

        # Normalize model name
        self.model_name = model_name.lower()

        # Initialize LLM based on model type
        if 'gpt' in self.model_name:
            self.llm = self._init_openai(api_key)
            self.provider = 'openai'
        elif 'claude' in self.model_name:
            self.llm = self._init_anthropic(api_key)
            self.provider = 'anthropic'
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _init_openai(self, api_key: Optional[str]) -> OpenAI:
        """Initialize OpenAI native client."""
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        model_config = self.judger_config['models']['gpt-4o']
        self.openai_model_name = model_config['model_name']
        self.temperature = self.judger_config['temperature']
        self.max_tokens = model_config.get('max_tokens', 2000)

        return OpenAI(api_key=api_key)

    def _init_anthropic(self, api_key: Optional[str]) -> Anthropic:
        """Initialize Anthropic native client."""
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        model_config = self.judger_config['models']['claude-3.5-sonnet']
        self.anthropic_model_name = model_config['model_name']
        self.temperature = self.judger_config['temperature']
        self.max_tokens = model_config.get('max_tokens', 2000)

        return Anthropic(api_key=api_key)

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 - EXACTLY matching WiserUI-Bench and Baseline method.

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

    def judge(
        self,
        spotter_output: Dict[str, Any],
        baseline_image_path: str
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Judge which UI variant is more effective based on baseline image and proposed changes.

        Args:
            spotter_output: JSON output from Spotter (baseline description + proposed changes)
            baseline_image_path: Path to baseline (first) UI image

        Returns:
            Tuple of (judgment, reasoning, metadata) where:
            - judgment: 'first' or 'second' (matching WiserUI-Bench format)
            - reasoning: Structured reasoning dict
            - metadata: API call metadata (tokens, latency, etc.)
        """
        # Format spotter output as string for prompt
        spotter_str = json.dumps(spotter_output, indent=2)

        # Get prompt
        prompts = format_judger_prompt(
            spotter_output=spotter_str
        )

        # Encode baseline image
        baseline_image_b64 = self._encode_image(baseline_image_path)

        # Call LLM based on provider
        start_time = time.time()

        try:
            if self.provider == 'openai':
                # OpenAI native API call
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompts['prompt']},
                            {"type": "image_url", "image_url": {"url": baseline_image_b64}}
                        ]
                    }
                ]

                response = self.llm.chat.completions.create(
                    model=self.openai_model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            elif self.provider == 'anthropic':
                # Anthropic native API call
                # Extract base64 data from data URI
                b64_data = baseline_image_b64.split(',')[1] if ',' in baseline_image_b64 else baseline_image_b64

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
                                    "data": b64_data
                                }
                            }
                        ]
                    }
                ]

                response = self.llm.messages.create(
                    model=self.anthropic_model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages
                )

                content = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            latency = time.time() - start_time

            metadata = {
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "model": self.model_name,
                "provider": self.provider
            }

            # Parse response
            try:
                reasoning = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                reasoning = {"raw_response": content}

            # Extract judgment using parse function
            judgment = parse_judger_response(content)

            metadata['prompt'] = prompts
            metadata['spotter_output'] = spotter_output

            return judgment, reasoning, metadata

        except Exception as e:
            raise RuntimeError(f"Judger failed: {str(e)}") from e
