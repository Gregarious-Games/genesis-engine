"""
Entrainment Layers
==================

D-module to T/Q module phase coupling using Kuramoto dynamics.

The stable D-modules act as master clocks, entraining the
constrained T/Q modules through phase coupling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

from ..core.constants import GAMMA, DYNAMICS


class KuramotoCoupling(nn.Module):
    """
    Kuramoto-style phase coupling.

    Implements: dθ/dt = ω + K * sin(θ_driver - θ)

    Where:
    - θ is the node's phase
    - ω is natural frequency
    - K is coupling strength
    - θ_driver is the driving phase (from D-module)
    """

    def __init__(self,
                 n_nodes: int,
                 coupling_strength: float = None):
        super().__init__()

        self.n_nodes = n_nodes
        self.K = coupling_strength or GAMMA  # GAMMA proven optimal

        # Natural frequencies (slight variation prevents lockstep)
        self.register_buffer(
            'omega',
            torch.ones(n_nodes) + 0.01 * torch.randn(n_nodes)
        )

    def forward(self,
                phases: torch.Tensor,
                driver_phase: torch.Tensor,
                dt: float = 0.01) -> torch.Tensor:
        """
        Compute phase update.

        Args:
            phases: Current node phases [n_nodes]
            driver_phase: Driving phase from D-module (scalar)
            dt: Time step

        Returns:
            new_phases: Updated phases [n_nodes]
        """
        # Phase difference
        phase_diff = driver_phase - phases

        # Kuramoto coupling force
        coupling_force = self.K * torch.sin(phase_diff)

        # Phase update
        d_phase = self.omega + coupling_force
        new_phases = phases + d_phase * dt

        # Wrap to [0, 2π]
        new_phases = new_phases % (2 * np.pi)

        return new_phases

    def compute_order_parameter(self, phases: torch.Tensor) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter (synchronization measure).

        Returns:
            r: Coherence [0, 1] where 1 = fully synchronized
            psi: Mean phase
        """
        z = torch.mean(torch.exp(1j * phases.to(torch.complex64)))
        r = torch.abs(z).item()
        psi = torch.angle(z).item()
        return r, psi


class EntrainmentLayer(nn.Module):
    """
    Full entrainment layer for D → T → Q coupling.

    Manages phase relationships between module types:
    - D-modules provide master phase
    - T-modules are entrained to D
    - Q-modules are entrained to D (optionally also T)
    """

    def __init__(self,
                 n_d: int = 2,
                 n_t: int = 3,
                 n_q: int = 4,
                 coupling_strength: float = None):
        super().__init__()

        self.n_d = n_d
        self.n_t = n_t
        self.n_q = n_q
        self.K = coupling_strength or GAMMA

        # Kuramoto couplers for T and Q
        self.t_coupler = KuramotoCoupling(n_t, self.K)
        self.q_coupler = KuramotoCoupling(n_q, self.K)

        # Cross-module coupling weights
        self.dt_weight = nn.Parameter(torch.tensor(1.0))
        self.dq_weight = nn.Parameter(torch.tensor(1.0))
        self.tq_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self,
                d_phases: torch.Tensor,
                t_phases: torch.Tensor,
                q_phases: torch.Tensor,
                dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply entrainment to T and Q modules.

        D-modules are NOT modified (they are the master clock).

        Args:
            d_phases: D-module phases [n_d]
            t_phases: T-module phases [n_t]
            q_phases: Q-module phases [n_q]
            dt: Time step

        Returns:
            new_t_phases: Updated T phases
            new_q_phases: Updated Q phases
        """
        # D-module mean phase (master clock)
        d_mean = torch.angle(torch.mean(torch.exp(1j * d_phases.to(torch.complex64))))
        d_mean = d_mean.real

        # T entrainment (from D)
        new_t_phases = self.t_coupler(
            t_phases,
            d_mean * self.dt_weight,
            dt
        )

        # Q entrainment (from D, optionally modulated by T)
        t_mean = torch.angle(torch.mean(torch.exp(1j * new_t_phases.to(torch.complex64))))
        t_mean = t_mean.real

        q_driver = d_mean * self.dq_weight + t_mean * self.tq_weight
        new_q_phases = self.q_coupler(
            q_phases,
            q_driver,
            dt
        )

        return new_t_phases, new_q_phases

    def get_coherence(self,
                      d_phases: torch.Tensor,
                      t_phases: torch.Tensor,
                      q_phases: torch.Tensor) -> dict:
        """
        Compute coherence metrics across modules.
        """
        all_phases = torch.cat([d_phases, t_phases, q_phases])
        r_global, psi_global = self.t_coupler.compute_order_parameter(all_phases)

        r_d, _ = self.t_coupler.compute_order_parameter(d_phases)
        r_t, _ = self.t_coupler.compute_order_parameter(t_phases)
        r_q, _ = self.t_coupler.compute_order_parameter(q_phases)

        return {
            'global': r_global,
            'd': r_d,
            't': r_t,
            'q': r_q,
        }
