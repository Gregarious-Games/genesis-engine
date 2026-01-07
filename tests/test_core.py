"""
Core Module Tests
=================

Tests for fundamental Genesis Engine components.
"""

import torch
import numpy as np
import sys
import os

# Add parent to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import PHI, GAMMA, KOIDE, verify_identities
from core.clamping import clamp, get_max_abs
from core.dennis_node import DennisNode
from core.module_types import DualityModule, TrinityModule, QuadraticModule, GenesisHemisphere


def test_constants():
    """Test fundamental constant relationships."""
    print("Testing constants...")

    # PHI identity
    assert abs(PHI**2 - PHI - 1) < 1e-15, "PHI^2 = PHI + 1 failed"

    # GAMMA identity
    assert abs(GAMMA - 1/(6*PHI)) < 1e-15, "GAMMA = 1/(6*PHI) failed"

    # KOIDE identity
    assert abs(KOIDE - 2/3) < 1e-15, "KOIDE = 2/3 failed"

    # 6*PHI*GAMMA = 1
    assert abs(6*PHI*GAMMA - 1) < 1e-15, "6*PHI*GAMMA = 1 failed"

    # Verify identities function
    assert verify_identities() == True

    print("  [PASS] All constant identities verified")


def test_clamping():
    """Test clamping utilities."""
    print("Testing clamping...")

    # Tensor clamping
    x = torch.tensor([0.5, -0.5, 1.5, -1.5])
    clamped = clamp(x, 1.0)
    assert torch.all(clamped <= 1.0), "Upper clamp failed"
    assert torch.all(clamped >= -1.0), "Lower clamp failed"

    # NaN handling
    x_nan = torch.tensor([1.0, float('nan'), float('inf')])
    clamped_nan = clamp(x_nan, 1.0)
    assert torch.all(torch.isfinite(clamped_nan)), "NaN handling failed"

    # Unbounded clamping
    x = torch.tensor([1000.0])
    clamped_inf = clamp(x, float('inf'))
    assert clamped_inf[0] == 1000.0, "Infinite clamp failed"

    # get_max_abs
    assert get_max_abs('D') == float('inf'), "D max_abs failed"
    assert get_max_abs('T') == GAMMA, "T max_abs failed"
    assert get_max_abs('Q') == GAMMA, "Q max_abs failed"

    print("  [PASS] Clamping utilities verified")


def test_dennis_node():
    """Test DennisNode oscillator."""
    print("Testing DennisNode...")

    # Create nodes of each type
    d_node = DennisNode('D', 'L', 'D1_L')
    t_node = DennisNode('T', 'L', 'T1_L')
    q_node = DennisNode('Q', 'R', 'Q1_R')

    # Check max_abs
    assert d_node.max_abs == float('inf'), "D node max_abs wrong"
    assert t_node.max_abs == GAMMA, "T node max_abs wrong"
    assert q_node.max_abs == GAMMA, "Q node max_abs wrong"

    # Run some steps
    for _ in range(100):
        z_neighbor = torch.tensor(0.01)
        d_out = d_node(z_neighbor)
        t_out = t_node(z_neighbor)
        q_out = q_node(z_neighbor)

    # Check outputs are finite
    assert torch.isfinite(d_out), "D output not finite"
    assert torch.isfinite(t_out), "T output not finite"
    assert torch.isfinite(q_out), "Q output not finite"

    # Check T/Q are constrained
    assert torch.abs(t_out) <= GAMMA, f"T output {t_out} exceeds GAMMA"
    assert torch.abs(q_out) <= GAMMA, f"Q output {q_out} exceeds GAMMA"

    # Check phases are computed
    assert torch.isfinite(d_node.phase), "D phase not finite"
    assert torch.isfinite(t_node.phase), "T phase not finite"

    print("  [PASS] DennisNode verified")


def test_modules():
    """Test module types."""
    print("Testing module types...")

    # Duality module
    d_mod = DualityModule('L')
    for _ in range(100):
        d_out = d_mod()
    assert d_out.shape == (2,), f"Duality output shape wrong: {d_out.shape}"
    assert torch.all(torch.isfinite(d_out)), "Duality output not finite"

    # Trinity module
    t_mod = TrinityModule('L')
    d_phase = d_mod.get_mean_phase()
    for _ in range(100):
        t_out = t_mod(d_phase=d_phase)
    assert t_out.shape == (3,), f"Trinity output shape wrong: {t_out.shape}"
    assert torch.all(torch.isfinite(t_out)), "Trinity output not finite"
    assert torch.all(torch.abs(t_out) <= GAMMA), "Trinity exceeds GAMMA"

    # Quadratic module
    q_mod = QuadraticModule('R')
    for _ in range(100):
        q_out = q_mod(d_phase=d_phase)
    assert q_out.shape == (4,), f"Quadratic output shape wrong: {q_out.shape}"
    assert torch.all(torch.isfinite(q_out)), "Quadratic output not finite"
    assert torch.all(torch.abs(q_out) <= GAMMA), "Quadratic exceeds GAMMA"

    print("  [PASS] Module types verified")


def test_hemisphere():
    """Test full hemisphere."""
    print("Testing GenesisHemisphere...")

    hemi = GenesisHemisphere('L')

    # Run 1000 steps
    for _ in range(1000):
        output = hemi()

    # Check structure
    assert 'd' in output, "Missing D output"
    assert 't' in output, "Missing T output"
    assert 'q' in output, "Missing Q output"

    # Check shapes
    assert output['d'].shape == (2,), f"D shape wrong: {output['d'].shape}"
    assert output['t'].shape == (3,), f"T shape wrong: {output['t'].shape}"
    assert output['q'].shape == (4,), f"Q shape wrong: {output['q'].shape}"

    # Check stability
    all_z = hemi.get_all_z()
    assert all_z.shape == (9,), f"All Z shape wrong: {all_z.shape}"
    assert torch.all(torch.isfinite(all_z)), "Some Z values not finite"

    # Check T/Q constraints
    t_vals = output['t']
    q_vals = output['q']
    assert torch.all(torch.abs(t_vals) <= GAMMA + 1e-6), f"T exceeds GAMMA: {t_vals}"
    assert torch.all(torch.abs(q_vals) <= GAMMA + 1e-6), f"Q exceeds GAMMA: {q_vals}"

    print("  [PASS] GenesisHemisphere verified")


def test_stability():
    """Test long-term stability (mini version of Lorenz test)."""
    print("Testing stability (10,000 steps)...")

    hemi = GenesisHemisphere('L')

    nan_count = 0
    max_d = 0
    max_t = 0
    max_q = 0

    for i in range(10000):
        # Random input
        noise = torch.randn(1) * 0.01
        output = hemi(noise)

        # Check for NaN
        all_z = hemi.get_all_z()
        if not torch.all(torch.isfinite(all_z)):
            nan_count += 1
            break

        # Track maxes
        max_d = max(max_d, output['d'].abs().max().item())
        max_t = max(max_t, output['t'].abs().max().item())
        max_q = max(max_q, output['q'].abs().max().item())

    assert nan_count == 0, f"NaN detected at step {i}"
    assert max_t <= GAMMA + 1e-6, f"T exceeded GAMMA: {max_t}"
    assert max_q <= GAMMA + 1e-6, f"Q exceeded GAMMA: {max_q}"

    print(f"  Max values - D: {max_d:.4f}, T: {max_t:.6f}, Q: {max_q:.6f}")
    print("  [PASS] Stability verified (10,000 steps)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GENESIS NN - CORE MODULE TESTS")
    print("=" * 60)

    test_constants()
    test_clamping()
    test_dennis_node()
    test_modules()
    test_hemisphere()
    test_stability()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
