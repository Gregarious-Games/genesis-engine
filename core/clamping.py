"""
Clamping Utilities
==================

Node-specific clamping strategies for Genesis architecture.

D-modules: Unbounded (self-regulating through dynamics)
T-modules: Clamped to GAMMA
Q-modules: Clamped to GAMMA
"""

import torch
import numpy as np
from typing import Union

from .constants import GAMMA, THRESHOLDS


def clamp(x: Union[torch.Tensor, np.ndarray, float],
          max_abs: float) -> Union[torch.Tensor, np.ndarray, float]:
    """
    Clamp values to [-max_abs, max_abs] with NaN protection.

    Args:
        x: Input tensor, array, or scalar
        max_abs: Maximum absolute value

    Returns:
        Clamped values with NaN replaced by 0
    """
    if isinstance(x, torch.Tensor):
        # Replace NaN/Inf with 0
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if max_abs == float('inf'):
            return x
        return torch.clamp(x, -max_abs, max_abs)

    elif isinstance(x, np.ndarray):
        x = np.where(np.isfinite(x), x, 0.0)
        if max_abs == float('inf'):
            return x
        return np.clip(x, -max_abs, max_abs)

    else:  # scalar
        if not np.isfinite(x):
            return 0.0
        if max_abs == float('inf'):
            return x
        return max(min(x, max_abs), -max_abs)


def get_max_abs(node_type: str) -> float:
    """
    Get max_abs constraint for a node type.

    Args:
        node_type: 'D' (duality), 'T' (trinity), or 'Q' (quadratic)

    Returns:
        Maximum absolute value for clamping
    """
    if node_type.upper() == 'D':
        return THRESHOLDS['d_max_abs']  # inf
    elif node_type.upper() == 'T':
        return THRESHOLDS['t_max_abs']  # GAMMA
    elif node_type.upper() == 'Q':
        return THRESHOLDS['q_max_abs']  # GAMMA
    else:
        raise ValueError(f"Unknown node type: {node_type}. Use 'D', 'T', or 'Q'.")


def soft_clamp(x: torch.Tensor, max_abs: float, k: float = None) -> torch.Tensor:
    """
    Soft clamping using tanh for differentiable constraint.

    Args:
        x: Input tensor
        max_abs: Target saturation value
        k: Steepness (default: 6*PHI from Genesis constants)

    Returns:
        Soft-clamped tensor approaching ±max_abs asymptotically
    """
    if k is None:
        k = THRESHOLDS['sigmoid_k']

    if max_abs == float('inf'):
        return x

    return max_abs * torch.tanh(x / max_abs)


def hysteresis_clamp(x: torch.Tensor,
                     max_abs: float,
                     state: torch.Tensor) -> tuple:
    """
    Clamping with hysteresis to prevent oscillation at boundaries.

    Args:
        x: Input tensor
        max_abs: Maximum absolute value
        state: Previous clamp state (0=normal, 1=clamped_high, -1=clamped_low)

    Returns:
        (clamped_x, new_state)
    """
    band = THRESHOLDS['hysteresis']
    new_state = state.clone()

    # Upper hysteresis
    hit_upper = x >= max_abs
    release_upper = x < max_abs - band
    new_state = torch.where(hit_upper, torch.ones_like(state), new_state)
    new_state = torch.where(release_upper & (state == 1),
                            torch.zeros_like(state), new_state)

    # Lower hysteresis
    hit_lower = x <= -max_abs
    release_lower = x > -max_abs + band
    new_state = torch.where(hit_lower, -torch.ones_like(state), new_state)
    new_state = torch.where(release_lower & (state == -1),
                            torch.zeros_like(state), new_state)

    # Apply clamp based on state
    clamped = x.clone()
    clamped = torch.where(new_state == 1,
                          torch.full_like(x, max_abs), clamped)
    clamped = torch.where(new_state == -1,
                          torch.full_like(x, -max_abs), clamped)

    return clamped, new_state
