# Spot & Judge UI/UX Evaluation Architecture

Decomposed UI/UX Evaluation Architecture using GLM-4.5V's Visual Perception to Mitigate Position Bias in MLLMs

## Overview

This research proposes a two-stage "Spot & Judge" architecture to mitigate Position Bias in Multimodal Large Language Models (MLLMs) when evaluating UI/UX designs. The architecture separates visual perception (Stage 1: Spotter) from UX reasoning (Stage 2: Judger) to reduce position-dependent bias.

## Key Features

- **Stage 1 - Spotter**: Uses GLM-4.5V to extract visual differences between UI variants in a position-agnostic manner
- **Stage 2 - Judger**: Uses GPT-4o or Claude 3.5 Sonnet to evaluate UX effectiveness based on extracted differences
- **Position Bias Mitigation**: Reduces position dependency by preventing simultaneous image comparison
- **WiserUI-Bench Integration**: Complete evaluation framework on real A/B test data

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/spot-judge-ui-evaluation.git
cd spot-judge-ui-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - ZAI_API_KEY
```

## Quick Start

```bash
# Run a small pilot experiment
python scripts/run_pilot.py

# Run full experiment
python scripts/run_experiment.py --num-samples 100 --models gpt-4o claude-3.5-sonnet
```

## Project Structure

```
spot-judge-ui-evaluation/
├── config/              # Experiment and model configurations
├── src/                 # Source code
│   ├── models/          # Spotter, Judger, Baseline models
│   ├── experiments/     # Experiment orchestration
│   ├── data/            # Dataset loading
│   └── utils/           # Utilities
├── scripts/             # Execution scripts
├── cache/               # Dataset cache (gitignored)
├── checkpoints/         # Experiment checkpoints (gitignored)
├── logs/                # Experiment logs (gitignored)
└── results/             # Results (gitignored)
```

## Configuration

Edit `config/experiment_config.yaml` and `config/models_config.yaml` to customize experiment settings.

## Results

Experiment results are saved in `results/` directory:
- `results/raw/`: Raw experiment results (JSON)
- `results/processed/`: Aggregated metrics (CSV, JSON)
- `results/visualizations/`: Visualization outputs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{spotjudge2025,
  title={Decomposed UI/UX Evaluation Architecture using GLM-4.5V's Visual Perception},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/spot-judge-ui-evaluation}
}
```

## License

[Add your license here]

## Contact

[Add your contact information here]

