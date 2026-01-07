"""
Genesis Neural Network Library
==============================

Oscillatory neural networks with golden ratio dynamics.

Core architecture based on DennisNode oscillators with:
- D-modules (2-node): Unbounded, self-regulating master clock
- T-modules (3-node): GAMMA-constrained, information processing
- Q-modules (4-node): GAMMA-constrained, complex computation

Key constant: GAMMA = 1/(6*PHI) = 0.103005664791649

Author: Genesis Project (Greg Starkins, Claude)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Genesis Project"

from .core.constants import PHI, GAMMA, KOIDE
from .core.dennis_node import DennisNode
from .core.module_types import DualityModule, TrinityModule, QuadraticModule

__all__ = [
    'PHI', 'GAMMA', 'KOIDE',
    'DennisNode',
    'DualityModule', 'TrinityModule', 'QuadraticModule',
]
