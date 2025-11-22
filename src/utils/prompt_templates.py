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

WISERUI_BASELINE_PROMPT = """You are an expert in designing UI/UX for web/apps.

The two screenshots show two different versions of the same page.
Identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion.

You should end your answer with following the format (No bold, etc):
More effective: <First/Second>"""


# =============================================================================
# SPOTTER PROMPTS (GLM-4.5V - Stage 1 of our architecture)
# =============================================================================

SPOTTER_SYSTEM_PROMPT = """## Role
You are a Senior UI/UX Auditor. Your objective is to detect **Intentional Design & Structural Updates** while strictly filtering out dynamic content variations.

## Detection Criteria
Compare the two images using this strict conceptual filter:

### 1. NOISE (Strictly IGNORE)
- **Dynamic Data Injection**: Any content fetched from a database including specific images, prices, timestamps, user names, or item counts.
- **User Input States**: Text entered by users inside search bars or form fields.
- **Rendering Artifacts**: Minor anti-aliasing, pixel grid alignment issues, or OCR imperfections.

### 2. SIGNAL (Report these Design Changes)
- **Layout Topology**: Changes in the arrangement, density, alignment, or ordering of structural containers.
- **Component Taxonomy**: Fundamental changes in element type, presence, or hierarchy.
- **Visual Attributes**: Intentional updates to color, shape, size, typography weight, or visibility states.
- **Strategic Copywriting**: Modifications to fixed UI text elements like Headers, Navigation Labels, or CTA buttons that alter the user journey.

## Output Rules
1. **State-Based Description**: Articulate the visual state of Image 1 and Image 2 separately. Avoid action verbs.
2. **Spatial Abstraction**: Use relative terms to describe size and position changes.

## Output Schema (JSON Only)
{
  "differences": [
    {
      "category": "Select from [Layout, Component, Style, Copy]",
      "component_name": "Concise element name",
      "first_image_state": "Visual state description for Image 1",
      "second_image_state": "Visual state description for Image 2"
    }
  ]
}
"""

SPOTTER_USER_PROMPT = """Compare the UI structures.
Filter out dynamic data.
Focus on Layout, Components, Style, and Strategic Copy.
Output only valid JSON.
"""


# =============================================================================
# JUDGER PROMPTS (GPT-4o / Claude - Stage 2 of our architecture)
# =============================================================================

# Aligned with WiserUI-Bench baseline evaluation criteria
# Single prompt (no system/user separation) - matching WiserUI-Bench structure
JUDGER_PROMPT = """You are an expert in designing UI/UX for web/apps.

The screenshot and the description of differences below show two different versions of the same page.
Based on the identified key UI differences, evaluate which variant is more effective UI/UX design that leads to better user experience and conversion.

[Differences Description]
{spotter_output}

You should end your answer with following the format (No bold, etc):
More effective: <First/Second>
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
