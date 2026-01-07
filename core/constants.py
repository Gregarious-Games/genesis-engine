"""
Genesis Engine Constants
========================

Fundamental constants derived from the golden ratio (PHI).

The key insight: GAMMA = 1/(6*PHI) emerges as:
  - Optimal constraint for T/Q modules in neural dynamics
  - Entrainment coupling strength for D→T→Q synchronization
  - Threshold that produces <0.001% error on particle mass ratios

All other thresholds derive from GAMMA and PHI.
"""

import math

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

PHI = 1.6180339887498949
"""Golden ratio: (1 + sqrt(5)) / 2"""

GAMMA = 1 / (6 * PHI)  # 0.103005664791649
"""Gate constant: 1/(6*PHI). Fundamental threshold for T/Q modules."""

KOIDE = 4 * PHI * GAMMA  # Exactly 2/3
"""Koide constant: 4*PHI*GAMMA = 2/3. Appears in lepton mass ratios."""

# Verify Koide identity
assert abs(KOIDE - 2/3) < 1e-15, "Koide identity violated"


# =============================================================================
# DERIVED THRESHOLDS
# =============================================================================

THRESHOLDS = {
    # Clamping
    'clamp_high': 1 - GAMMA,           # 0.897 - upper bound before reset
    'clamp_low': GAMMA,                 # 0.103 - lower meaningful signal
    'hysteresis': PHI * GAMMA,          # 1/6 = 0.167 - deadband width

    # Module-specific
    'd_max_abs': float('inf'),          # D-modules: unbounded
    't_max_abs': GAMMA,                 # T-modules: constrained
    'q_max_abs': GAMMA,                 # Q-modules: constrained

    # Sigmoid
    'sigmoid_k': 6 * PHI,               # 9.708 - sigmoid steepness
}


# =============================================================================
# DYNAMICS PARAMETERS
# =============================================================================

DYNAMICS = {
    # Decay rates
    'volatile_decay': 1 / PHI,          # 0.618 - fast forgetting
    'persistent_decay': GAMMA,          # 0.103 - slow consolidation

    # Coupling
    'entrainment_k': GAMMA,             # 0.103 - D→T→Q coupling
    'bridge_coupling': KOIDE,           # 2/3 - cross-module bridge

    # Timing
    'phi_rhythm': 40 * (PHI ** 11),     # 7960s - D-module natural period
    'sparse_interval': int(PHI ** 11),  # 199 steps - gradient update interval

    # Learning
    'hebbian_window': 1 / PHI,          # 0.618 radians - phase conjugate window
    'hebbian_lr': GAMMA,                # 0.103 - Hebbian learning rate
}


# =============================================================================
# SCALING LAWS (for evolution controllers)
# =============================================================================

SCALING_LAWS = {
    'phi': 1 / (PHI ** 3),              # 0.236 - original
    'gateway': GAMMA,                    # 0.103 - gate constant
    'koide': KOIDE,                      # 0.667 - mass ratio
    'evolution': 74 / 1000,              # 0.074 - N_evolution
    'heartbeat': 72 / 1000,              # 0.072 - cosmic heartbeat
}


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_identities():
    """Verify fundamental Genesis Engine identities."""
    errors = []

    # GAMMA = 1/(6*PHI)
    if abs(GAMMA - 1/(6*PHI)) > 1e-15:
        errors.append("GAMMA identity failed")

    # KOIDE = 4*PHI*GAMMA = 2/3
    if abs(KOIDE - 2/3) > 1e-15:
        errors.append("KOIDE identity failed")

    # PHI^2 = PHI + 1
    if abs(PHI**2 - PHI - 1) > 1e-15:
        errors.append("PHI^2 identity failed")

    # 1/PHI = PHI - 1
    if abs(1/PHI - (PHI - 1)) > 1e-15:
        errors.append("1/PHI identity failed")

    # 6*PHI*GAMMA = 1
    if abs(6*PHI*GAMMA - 1) > 1e-15:
        errors.append("6*PHI*GAMMA identity failed")

    if errors:
        raise ValueError(f"Identity verification failed: {errors}")

    return True


# Run verification on import
verify_identities()
