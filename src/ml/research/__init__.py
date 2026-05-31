"""
src/ml/research/

Model-research loop tooling (see docs/MODEL_RESEARCH_AGENT.md). This package
holds the mechanical harness the model-research agent calls to turn a
hypothesis into a shadow-testable challenger model. It is deliberately separate
from src/ml/train.py (the champion training pipeline) so research experiments
never mutate the frozen champion path.
"""
