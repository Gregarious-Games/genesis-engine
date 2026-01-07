"""Genesis NN Core Module"""

from .constants import PHI, GAMMA, KOIDE, THRESHOLDS, DYNAMICS
from .dennis_node import DennisNode
from .module_types import DualityModule, TrinityModule, QuadraticModule
from .clamping import clamp, get_max_abs

__all__ = [
    'PHI', 'GAMMA', 'KOIDE', 'THRESHOLDS', 'DYNAMICS',
    'DennisNode',
    'DualityModule', 'TrinityModule', 'QuadraticModule',
    'clamp', 'get_max_abs',
]
