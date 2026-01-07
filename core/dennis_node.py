"""
DennisNode - Core Oscillator Unit
=================================

The fundamental building block of Genesis neural networks.

Each DennisNode is a coupled oscillator with:
- Internal state cascade (X1 → X2 → X3 → phi → Y → Z)
- Energy accumulation and dissipation
- Phase tracking for Hebbian learning
- Type-specific clamping (D=unbounded, T/Q=GAMMA)

This is a HYBRID implementation:
- Forward pass is differentiable (for sparse gradient updates)
- Internal state updates are in-place (like biological neurons)
- Hebbian learning is separate from backprop
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple

from .constants import PHI, GAMMA, DYNAMICS
from .clamping import clamp, get_max_abs


class DennisNode(nn.Module):
    """
    Single oscillator node with Genesis Engine dynamics.

    The node maintains internal state that evolves according to:
        X1 += forward_rate - center_pull * X1
        X2 = X1 - reverse_rate
        phi = 0.5 * (X1 + X2)
        X3 = phi * (1 - damping)
        Y = |X3 - phi|
        Z = clamp(X3 * Y + coupling + noise, max_abs)

    Args:
        node_type: 'D' (duality), 'T' (trinity), or 'Q' (quadratic)
        hemisphere: 'L' (left) or 'R' (right)
        name: Optional identifier for this node

    Attributes:
        Z: Current output value
        phase: Current phase (for Hebbian learning)
        health: Node health metric [0, 1]
        energy: Accumulated energy
    """

    def __init__(self,
                 node_type: str = 'D',
                 hemisphere: str = 'L',
                 name: Optional[str] = None):
        super().__init__()

        self.node_type = node_type.upper()
        self.hemisphere = hemisphere.upper()
        self.name = name or f"{self.node_type}_{self.hemisphere}"
        self.max_abs = get_max_abs(self.node_type)

        # === LEARNABLE PARAMETERS (updated via sparse gradients) ===
        self.coupling_base = nn.Parameter(torch.tensor(0.05))
        self.internal_forward = nn.Parameter(torch.tensor(0.02))
        self.internal_reverse = nn.Parameter(torch.tensor(0.01))
        self.center_pull = nn.Parameter(torch.tensor(0.3))
        self.damping = nn.Parameter(torch.tensor(0.01))
        self.noise_amplitude = nn.Parameter(torch.tensor(0.001))

        # === INTERNAL STATE (non-differentiable, updated in-place) ===
        self.register_buffer('X1', torch.tensor(0.0))
        self.register_buffer('X2', torch.tensor(0.0))
        self.register_buffer('X3', torch.tensor(0.0))
        self.register_buffer('phi_internal', torch.tensor(0.0))
        self.register_buffer('Y', torch.tensor(0.0))
        self.register_buffer('Z', torch.tensor(0.0))
        self.register_buffer('energy', torch.tensor(0.01))
        self.register_buffer('phase', torch.tensor(0.0))
        self.register_buffer('health', torch.tensor(1.0))

        # State history for phase computation
        self.register_buffer('z_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))

    def reset_state(self):
        """Reset all internal state to initial values."""
        self.X1.zero_()
        self.X2.zero_()
        self.X3.zero_()
        self.phi_internal.zero_()
        self.Y.zero_()
        self.Z.zero_()
        self.energy.fill_(0.01)
        self.phase.zero_()
        self.health.fill_(1.0)
        self.z_history.zero_()
        self.history_idx.zero_()

    def forward(self,
                z_neighbors: torch.Tensor,
                threshold: float = 0.01,
                entrainment_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute one timestep of oscillator dynamics.

        Args:
            z_neighbors: Mean Z value of neighboring nodes
            threshold: Tension threshold for coupling sign flip
            entrainment_phase: Optional phase from D-module for T/Q entrainment

        Returns:
            Z: Current output value (clamped for T/Q, unbounded for D)
        """
        # === ENERGY DYNAMICS ===
        tension = torch.abs(z_neighbors - self.Z)
        self.energy = self.energy * 0.98 + tension * 100
        self.energy = torch.clamp(self.energy, 0, 1e6)

        # === COUPLING ===
        coupling = torch.where(
            tension < threshold,
            -self.coupling_base,
            self.coupling_base
        )

        # === ENTRAINMENT (for T/Q modules) ===
        if entrainment_phase is not None and self.node_type != 'D':
            phase_diff = entrainment_phase - self.phase
            entrainment_force = GAMMA * torch.sin(phase_diff)
            coupling = coupling + entrainment_force * 0.1

        # === INTERNAL CASCADE ===
        # Accumulate energy into X1
        energy_injection = self.energy * 0.005
        self.X1 = self.X1 + self.internal_forward + energy_injection
        self.X1 = self.X1 - self.center_pull * self.X1

        # Propagate through cascade
        self.X2 = self.X1 - self.internal_reverse
        self.phi_internal = 0.5 * (self.X1 + self.X2)
        self.X3 = self.phi_internal * (1 - self.damping)
        self.Y = torch.abs(self.X3 - self.phi_internal)

        # === OUTPUT ===
        noise = self.noise_amplitude * torch.randn(1).squeeze().to(self.Z.device)
        raw_z = self.X3 * self.Y + coupling * 0.1 + noise

        # Type-specific clamping
        if self.max_abs == float('inf'):
            self.Z = raw_z  # D-modules: unbounded
        else:
            self.Z = torch.clamp(raw_z, -self.max_abs, self.max_abs)

        # NaN protection
        if not torch.isfinite(self.Z):
            self.Z = torch.tensor(0.0, device=self.Z.device)

        # === PHASE UPDATE ===
        self._update_phase()

        # === HEALTH UPDATE ===
        self._update_health()

        return self.Z

    def _update_phase(self):
        """Update phase using Hilbert transform approximation."""
        # Store in circular buffer
        idx = self.history_idx.item()
        self.z_history[idx] = self.Z.detach()
        self.history_idx = (self.history_idx + 1) % 100

        # Simple phase estimate from recent values
        if idx >= 2:
            # Approximate derivative
            dz = self.z_history[idx] - self.z_history[idx - 1]
            z_current = self.z_history[idx]

            # Phase from quadrature
            self.phase = torch.atan2(dz, z_current + 1e-10)

    def _update_health(self):
        """Update node health metric."""
        # Health decreases with saturation, increases with activity
        if self.max_abs != float('inf'):
            saturation = torch.abs(self.Z) / self.max_abs
            self.health = self.health * 0.99 + (1 - saturation) * 0.01
        else:
            # D-modules: health based on energy stability
            energy_norm = torch.clamp(self.energy / 100, 0, 1)
            self.health = self.health * 0.99 + (1 - energy_norm * 0.5) * 0.01

        self.health = torch.clamp(self.health, 0, 1)

    def detect_saturation(self) -> bool:
        """Check if node is saturated at constraint boundary."""
        if self.max_abs == float('inf'):
            return False
        return torch.abs(self.Z) >= self.max_abs * 0.99

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state as dictionary."""
        return {
            'Z': self.Z.clone(),
            'phase': self.phase.clone(),
            'energy': self.energy.clone(),
            'health': self.health.clone(),
            'X1': self.X1.clone(),
            'X2': self.X2.clone(),
            'X3': self.X3.clone(),
        }

    def extra_repr(self) -> str:
        return f"type={self.node_type}, hemisphere={self.hemisphere}, max_abs={self.max_abs}"


class DennisNodeBatch(nn.Module):
    """
    Batched version of DennisNode for efficient processing.

    Processes multiple nodes of the same type in parallel.
    """

    def __init__(self,
                 n_nodes: int,
                 node_type: str = 'D',
                 hemisphere: str = 'L'):
        super().__init__()

        self.n_nodes = n_nodes
        self.node_type = node_type.upper()
        self.hemisphere = hemisphere.upper()
        self.max_abs = get_max_abs(self.node_type)

        # Learnable parameters (shared across batch)
        self.coupling_base = nn.Parameter(torch.full((n_nodes,), 0.05))
        self.internal_forward = nn.Parameter(torch.full((n_nodes,), 0.02))
        self.internal_reverse = nn.Parameter(torch.full((n_nodes,), 0.01))
        self.center_pull = nn.Parameter(torch.full((n_nodes,), 0.3))
        self.damping = nn.Parameter(torch.full((n_nodes,), 0.01))
        self.noise_amplitude = nn.Parameter(torch.full((n_nodes,), 0.001))

        # Internal state
        self.register_buffer('X1', torch.zeros(n_nodes))
        self.register_buffer('X2', torch.zeros(n_nodes))
        self.register_buffer('X3', torch.zeros(n_nodes))
        self.register_buffer('phi_internal', torch.zeros(n_nodes))
        self.register_buffer('Y', torch.zeros(n_nodes))
        self.register_buffer('Z', torch.zeros(n_nodes))
        self.register_buffer('energy', torch.full((n_nodes,), 0.01))
        self.register_buffer('phase', torch.zeros(n_nodes))
        self.register_buffer('health', torch.ones(n_nodes))

    def forward(self,
                z_neighbors: torch.Tensor,
                threshold: float = 0.01,
                entrainment_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process all nodes in parallel."""
        # Energy dynamics
        tension = torch.abs(z_neighbors - self.Z)
        self.energy = self.energy * 0.98 + tension * 100
        self.energy = torch.clamp(self.energy, 0, 1e6)

        # Coupling
        coupling = torch.where(
            tension < threshold,
            -self.coupling_base,
            self.coupling_base
        )

        # Entrainment
        if entrainment_phase is not None and self.node_type != 'D':
            phase_diff = entrainment_phase - self.phase
            entrainment_force = GAMMA * torch.sin(phase_diff)
            coupling = coupling + entrainment_force * 0.1

        # Internal cascade
        energy_injection = self.energy * 0.005
        self.X1 = self.X1 + self.internal_forward + energy_injection
        self.X1 = self.X1 - self.center_pull * self.X1
        self.X2 = self.X1 - self.internal_reverse
        self.phi_internal = 0.5 * (self.X1 + self.X2)
        self.X3 = self.phi_internal * (1 - self.damping)
        self.Y = torch.abs(self.X3 - self.phi_internal)

        # Output
        noise = self.noise_amplitude * torch.randn(self.n_nodes, device=self.Z.device)
        raw_z = self.X3 * self.Y + coupling * 0.1 + noise

        # Clamping
        if self.max_abs == float('inf'):
            self.Z = raw_z
        else:
            self.Z = torch.clamp(raw_z, -self.max_abs, self.max_abs)

        # NaN protection
        self.Z = torch.where(torch.isfinite(self.Z), self.Z, torch.zeros_like(self.Z))

        # Phase update (simplified)
        self.phase = torch.atan2(self.X2, self.X1 + 1e-10)

        return self.Z
