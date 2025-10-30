"""
GLM-4.5V Spotter Module - Stage 1 of Spot & Judge Architecture

This module implements the "Spotter" - a visual perception component that
uses GLM-4.5V to extract objective visual differences between UI variants.

Key features:
- Position-agnostic visual analysis
- Grounding capability for element localization
- Thinking mode for deep visual reasoning
- Structured JSON output
"""

import os
import json
import time
import base64
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests
import yaml
from PIL import Image
import io

from ..utils.prompt_templates import format_spotter_prompt


class GLM4VSpotter:
    """
    Spotter using GLM-4.5V for visual difference extraction.

    This is Stage 1 of the Spot & Judge architecture:
    Input: Two UI images
    Output: Position-agnostic JSON describing visual differences
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: str = "config/models_config.yaml"
    ):
        """
        Initialize GLM-4.5V Spotter.

        Args:
            api_key: Z.AI API key (or set ZAI_API_KEY env var)
            config_path: Path to model configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config['spotter']

        # API configuration
        self.api_key = api_key or os.getenv('ZAI_API_KEY')
        if not self.api_key:
            raise ValueError("ZAI_API_KEY not found in environment or constructor")

        self.api_endpoint = self.config['api_endpoint']
        self.model_name = self.config['model_name']

        # Model parameters
        self.temperature = self.config.get('temperature', 0.2)
        self.max_tokens = self.config.get('max_tokens', 8000)
        self.thinking_mode = self.config.get('thinking_mode', 'enabled')
        self.timeout = self.config.get('timeout', 120)

        # Retry configuration
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 10)  # Increased for rate limits

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string with data URI prefix
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Encode to base64
        encoded = base64.b64encode(image_data).decode('utf-8')

        # Determine MIME type
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/png')

        return f"data:{mime_type};base64,{encoded}"

    def spot_differences(
        self,
        image_a_path: str,
        image_b_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract visual differences between two UI designs.

        Args:
            image_a_path: Path to first UI image
            image_b_path: Path to second UI image

        Returns:
            Tuple of (parsed_json, metadata) where:
            - parsed_json: Structured differences as dict
            - metadata: API call metadata (tokens, latency, etc.)
        """
        # Encode images
        image_a_b64 = self.encode_image(image_a_path)
        image_b_b64 = self.encode_image(image_b_path)

        # Format prompt
        prompts = format_spotter_prompt()

        # Prepare API request
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": prompts['system']
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts['user']},
                        {"type": "image_url", "image_url": {"url": image_a_b64}},
                        {"type": "image_url", "image_url": {"url": image_b_b64}}
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": self.thinking_mode}
        }

        # Make API call with retry logic
        response, metadata = self._call_api_with_retry(payload)

        # Parse JSON response
        try:
            parsed_json = self._parse_response(response)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return raw response with error flag
            parsed_json = {
                "error": "JSON parse error",
                "raw_response": response,
                "error_detail": str(e)
            }

        metadata['prompt'] = prompts
        metadata['image_a'] = image_a_path
        metadata['image_b'] = image_b_path

        return parsed_json, metadata

    def _call_api_with_retry(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Call GLM-4.5V API with exponential backoff retry.

        Args:
            payload: API request payload

        Returns:
            Tuple of (response_text, metadata)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()

                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                latency = time.time() - start_time

                response.raise_for_status()

                # Parse response
                response_data = response.json()

                # Extract content and usage
                content = response_data['choices'][0]['message']['content']
                usage = response_data.get('usage', {})

                metadata = {
                    "latency_seconds": latency,
                    "input_tokens": usage.get('prompt_tokens', 0),
                    "output_tokens": usage.get('completion_tokens', 0),
                    "total_tokens": usage.get('total_tokens', 0),
                    "model": self.model_name,
                    "attempt": attempt + 1
                }

                return content, metadata

            except requests.exceptions.RequestException as e:
                last_exception = e

                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"API call failed (attempt {attempt + 1}), retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"API call failed after {self.retry_attempts} attempts: {str(e)}"
                    ) from last_exception

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse GLM-4.5V response to extract JSON.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON as dictionary
        """
        # Try to parse as direct JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to extract JSON from anywhere in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        # If all else fails, raise error
        raise json.JSONDecodeError(
            "Could not extract JSON from response",
            response,
            0
        )
