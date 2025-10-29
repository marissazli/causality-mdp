# BAD-ACTS Benchmark

## Introduction

BAD-ACTS (Benchmark of ADversarial ACTionS) is a comprehensive benchmark designed to evaluate the robustness of agentic systems against adversarial manipulation that may result in harmful or undesirable behavior. Introduced in the paper *"Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harmful Actions,"* BAD-ACTS provides a structured framework for analyzing how susceptible different types of agentic systems are to adversarial influence.

This repository includes the complete BAD-ACTS benchmark suite, featuring four distinct environments—Travel Planner, Financial Article Writing, Code Generation, and Multi-Agent Debate. It also contains a dataset of adversarial actions, implementations of adversarial and defensive agents, and scripts for reproducing the experiments described in the paper.

## Contents

* `environments/` — Source code for the four benchmark environments
* `dataset/BAD-ACTS.csv` — Dataset of harmful actions
* `agents/` — Implementations of adversarial and defensive agents
* `evaluation/` — Evaluation functions tailored to each environment
* `run_experiments.py` — Script for running individual experiments
* `scripts/` — Utilities for executing full-scale experiments across agents, models, and environments

## Installation

To get started, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

We recommend using Python 3.10 or later.

## Running Individual Experiments

Use the `run_experiments.py` script to run standalone experiments with customizable settings for the environment, model, and agent. The script accepts the following arguments:

* `--model-client` *(str)*: Specifies the model backend. Default is `llama3.1:70b`. Compatible with models hosted via [Ollama](https://ollama.com/) and OpenAI.
* `--environment` *(str)*: Selects the experiment environment. Options: `travel_planning`, `financial_article_writing`, `code_generation`, `multi_agent_debate`. Default is `travel_planning`.
* `--adversarial-agent` *(str)*: Specifies the adversarial agent implementation.
* `--safe` *(flag)*: If set, uses Adversarial Action Warning (AAW) prompts instead of the default.
* `--guardian` *(flag)*: If set, enables a guardian agent that monitors and may intervene.
* `--id` *(int)*: Unique identifier for the experiment, useful for tracking and logging.

These arguments allow for flexible, targeted evaluation of robustness in agentic systems.

### Example Usage

```bash
python run_experiments.py --model-client llama3.1:70b --environment travel_planning --adversarial-agent PLANNER_AGENT
```

## Running Full-Scale Experiments

To use GPT-based models (e.g., `gpt-4`, `gpt-3.5-turbo`) with BAD-ACTS, you must set your OpenAI API key as an environment variable before running experiments:

```bash
export OPENAI_API_KEY=your_api_key_here
```

To replicate experiments across all agents within a specific environment using a designated adversarial agent, use the scripts in the `scripts/` directory. These batch scripts support the same command-line arguments as `run_experiments.py`.

### Example Script Usage

```bash
bash scripts/gpt-4.1/run_travel_planner_experiments.sh --safe --id 1
```