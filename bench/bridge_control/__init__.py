"""BridgeControl: a fenced, non-gated causal coverage experiment.

This package is research apparatus, not a phase gate.  It deliberately lives
outside :mod:`bench.evals` and never changes ``bench/SHIPPED``.
"""

from .fixture import BridgeControlEnv, BridgeDataset, FactorCell, generate_dataset

__all__ = ["BridgeControlEnv", "BridgeDataset", "FactorCell", "generate_dataset"]
