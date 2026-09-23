# GSoC 2026: DeepLense AI Scientist

Agentic systems for automating strong gravitational lensing research under DeepLense.

Author: Aatmaj Amol Salunke

> **The code for this project does not live in this folder.** It is in
> [ML4SCI/DeepLense-AI-Scientist](https://github.com/ML4SCI/DeepLense-AI-Scientist),
> where it was merged in [PR #7](https://github.com/ML4SCI/DeepLense-AI-Scientist/pull/7).
> This folder is a pointer so the project is listed alongside the other DeepLense projects.

## What I built

Two systems, both built on Pydantic AI:

1. **AI Scientist closed-loop pipeline.** Designs, trains and tunes neural networks for
   dark matter substructure classification. An experiment planner runs an iterative beam
   search over six buildable architecture families, then tunes the best candidates with
   ReAct-style hyperparameter tuning.
2. **Simulation code-generation agent.** Turns natural-language physics descriptions into
   lenstronomy simulation code, which runs in a Docker sandbox. It extracts and validates
   the physical parameters first, generates code second, and then checks that the code
   actually uses the parameters that were asked for.

## Headline result

An agent-discovered architecture, `cnn_s4_mid_w224` with **1.19M parameters**, reached
**0.9264** held-out test accuracy, against **0.9066** for a hand-picked ResNet-34 baseline
with **21.28M parameters**. That is higher accuracy with **17.9x fewer parameters**. Both
accuracies are means over two seeds, and the gap held on each seed. Full numbers are in
[BUDGET_FAIRNESS.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/BUDGET_FAIRNESS.md).

## Links

- Code: [ML4SCI/DeepLense-AI-Scientist](https://github.com/ML4SCI/DeepLense-AI-Scientist)
- Merged pull request: [ML4SCI/DeepLense-AI-Scientist#7](https://github.com/ML4SCI/DeepLense-AI-Scientist/pull/7)
- Blog post: [An AI Scientist for Dark Matter: Closed-Loop Agents for Strong Gravitational Lensing](https://medium.com/@aatmajsalunke/an-ai-scientist-for-dark-matter-closed-loop-agents-for-strong-gravitational-lensing-e6e31a4b13ad)

### Key docs

| Doc | What it covers |
|---|---|
| [WORKFLOW_DESIGN.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/WORKFLOW_DESIGN.md) | Overall design of the pipeline |
| [BROADER_SEARCH_RESULTS.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/BROADER_SEARCH_RESULTS.md) | Beam search over six architecture families |
| [BUDGET_FAIRNESS.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/BUDGET_FAIRNESS.md) | Budget-fairness controls and the two-seed headline result |
| [MULTI_SEED_RESULTS.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/MULTI_SEED_RESULTS.md) | Multi-seed replication of the main experiments |
| [AGENT_RELIABILITY.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/AGENT_RELIABILITY.md) | How reliable the planner's and LLM judge's decisions are |
| [PARAMETER_EXTRACTION.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/PARAMETER_EXTRACTION.md) | Two-stage simulation path: parameter extraction and validation |
| [MODEL_COMPARISON.md](https://github.com/ML4SCI/DeepLense-AI-Scientist/blob/main/docs/MODEL_COMPARISON.md) | Code-generation model comparison for the simulation agent |
