"""
Prompt templates for Spot & Judge experiment.

Baseline prompts EXACTLY replicate WiserUI-Bench methodology.
Spot & Judge prompts implement our proposed 2-stage architecture.

Reference: https://arxiv.org/html/2505.05026v3
"""

# =============================================================================
# BASELINE E2E PROMPTS (EXACT replication of WiserUI-Bench)
# Source: https://github.com/jeochris/wiserui-bench/blob/main/inference/prompts_task1/zero_shot.txt
# =============================================================================

# Official WiserUI-Bench zero-shot prompt (single unified prompt, no system/user separation)
WISERUI_BASELINE_PROMPT = """You are an expert in designing UI/UX for web/apps.

The two screenshots show two different versions of the same page.
Identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion.

You should end your answer with following the format (No bold, etc):
More effective: <First/Second>"""


# =============================================================================
# SPOTTER PROMPTS (GLM-4.5V - Stage 1 of our architecture)
# =============================================================================

SPOTTER_SYSTEM_PROMPT = """You are a precise visual analyst specializing in UI/UX design comparison.

Your ONLY role is to objectively identify WHAT changed between two UI designs - the FIRST (baseline) and SECOND (variant).

CRITICAL RULES:
1. Describe ONLY what changed - do NOT judge, evaluate, or predict impact
2. Be concise and quantitative (e.g., "4 to 5 products per row" not detailed product names)
3. Focus on changes relevant to user experience (layout, content, visual elements)
4. Ignore minor details like typos or exact product names unless UX-significant
5. Output MUST be valid JSON with simple aspect + description structure

Remember: You are a camera, not a critic. Describe changes objectively."""

SPOTTER_USER_PROMPT = """Compare the two UI designs and list what changed from the FIRST (baseline) to SECOND (variant).

For each change, identify:
- Aspect: What UI element/area changed (e.g., "Grid layout", "Button style")
- Description: What changed objectively (be quantitative when possible)

Output format (JSON):
{{
  "changes": [
    {{
      "aspect": "Product grid density",
      "description": "Increased from 4 to 5 products per row (12 → 15+ total visible)"
    }},
    {{
      "aspect": "Product labels",
      "description": "Added 'More colours' tags to some products"
    }}
  ]
}}

Be concise and objective. Focus on UX-relevant changes only. Output only valid JSON."""


# =============================================================================
# JUDGER PROMPTS (GPT-4o / Claude - Stage 2 of our architecture)
# =============================================================================

# Aligned with WiserUI-Bench baseline evaluation criteria
# Single prompt (no system/user separation) - matching WiserUI-Bench structure
JUDGER_PROMPT = """You are an expert in designing UI/UX for web/apps.

A screenshot and changes description show two different versions of the same page.
Identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion.

You should end your answer with following the format (No bold, etc):
More effective: <First/Second>

changes: {spotter_output}
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_baseline_prompt(position: str = "first_second") -> dict:
    """
    Format baseline prompt exactly as in WiserUI-Bench.

    Official WiserUI-Bench uses a SINGLE unified prompt (no system/user separation).

    Args:
        position: "first_second" or "second_first" for permutation testing

    Returns:
        dict with 'prompt' key containing the single unified prompt
    """
    return {
        "prompt": WISERUI_BASELINE_PROMPT
    }


def format_spotter_prompt() -> dict:
    """
    Format Spotter prompt for GLM-4.5V.

    Returns:
        dict with 'system' and 'user' prompts
    """
    return {
        "system": SPOTTER_SYSTEM_PROMPT,
        "user": SPOTTER_USER_PROMPT
    }


def format_judger_prompt(spotter_output: str) -> dict:
    """
    Format Judger prompt for GPT-4o/Claude.

    Uses single prompt format aligned with WiserUI-Bench baseline.

    Args:
        spotter_output: JSON output from Spotter (baseline + proposed changes)

    Returns:
        dict with 'prompt' key containing the formatted prompt
    """
    prompt = JUDGER_PROMPT.format(spotter_output=spotter_output)
    return {"prompt": prompt}


def parse_baseline_response(response: str) -> str:
    """
    Parse baseline response to extract First/Second answer.
    Uses WiserUI-Bench's exact parsing method (split instead of regex).

    Expected format: "More effective: <First/Second>"

    Returns:
        "first" or "second" (lowercase)
    """
    # WiserUI-Bench uses split() method
    try:
        answer = response.split('More effective:')[1].strip()
        if 'first' in answer.lower():
            return 'first'
        elif 'second' in answer.lower():
            return 'second'
    except (IndexError, AttributeError):
        pass

    # Fallback: look for first/second anywhere in response
    response_lower = response.lower()
    if 'first' in response_lower:
        return 'first'
    if 'second' in response_lower:
        return 'second'

    raise ValueError(f"Could not parse baseline response: {response}")


def parse_judger_response(response: str) -> str:
    """
    Parse Judger response to extract decision.

    Expected format: "More effective: <First/Second>"

    Args:
        response: Response from Judger

    Returns:
        "first" if First (prefer baseline shown in screenshot)
        "second" if Second (prefer baseline with changes applied)
    """
    # Try parsing "More effective: <First/Second>" format (same as baseline)
    try:
        answer = response.split('More effective:')[1].strip()
        if 'first' in answer.lower():
            return 'first'
        elif 'second' in answer.lower():
            return 'second'
    except (IndexError, AttributeError):
        pass

    # Fallback: look for first/second anywhere in response
    response_lower = response.lower()
    if 'first' in response_lower and 'second' not in response_lower:
        return 'first'
    if 'second' in response_lower and 'first' not in response_lower:
        return 'second'

    raise ValueError(f"Could not parse judger response: {response[:200]}")
