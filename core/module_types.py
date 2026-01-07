"""
Module Types
============

Pre-configured node groupings following Genesis architecture:

- DualityModule (2 nodes): Unbounded, master clock
- TrinityModule (3 nodes): GAMMA-constrained, processing
- QuadraticModule (4 nodes): GAMMA-constrained, complex computation

Each module type has distinct stability properties derived from geometry.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .constants import PHI, GAMMA, DYNAMICS
from .dennis_node import DennisNode, DennisNodeBatch


class DualityModule(nn.Module):
    """
    2-node Duality Module (D-type).

    The master clock of the Genesis architecture.
    - Unbounded dynamics (self-regulating)
    - Natural period: 40 * PHI^11 seconds
    - Drives entrainment for T and Q modules

    The 2-node structure is intrinsically stable because:
    - Even-numbered phase relationships don't accumulate
    - Energy naturally oscillates between two poles
    """

    def __init__(self, hemisphere: str = 'L', name_prefix: str = 'D'):
        super().__init__()

        self.hemisphere = hemisphere.upper()
        self.n_nodes = 2

        # Create 2 D-nodes
        self.nodes = nn.ModuleList([
            DennisNode('D', hemisphere, f"{name_prefix}{i+1}_{hemisphere}")
            for i in range(2)
        ])

        # Inter-node coupling weight
        self.coupling = nn.Parameter(torch.tensor(0.1))

    def forward(self, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process one timestep.

        Args:
            external_input: Optional external driving signal

        Returns:
            Z values for both nodes [2]
        """
        # Get current Z values
        z_values = torch.stack([node.Z for node in self.nodes])

        # Compute neighbor means (each node's neighbor is the other)
        z_neighbors = torch.stack([z_values[1], z_values[0]])

        # Add external input if provided
        if external_input is not None:
            z_neighbors = z_neighbors + external_input * 0.1

        # Update each node
        outputs = []
        for i, node in enumerate(self.nodes):
            z_out = node(z_neighbors[i])
            outputs.append(z_out)

        return torch.stack(outputs)

    def get_mean_phase(self) -> torch.Tensor:
        """Get circular mean of node phases (for entrainment)."""
        phases = torch.stack([node.phase for node in self.nodes])
        # Circular mean via complex exponential
        mean_phase = torch.angle(torch.mean(torch.exp(1j * phases.to(torch.complex64))))
        return mean_phase.real

    def get_state(self) -> Dict:
        """Get module state."""
        return {
            'z': torch.stack([n.Z for n in self.nodes]),
            'phase': torch.stack([n.phase for n in self.nodes]),
            'energy': torch.stack([n.energy for n in self.nodes]),
            'health': torch.stack([n.health for n in self.nodes]),
        }


class TrinityModule(nn.Module):
    """
    3-node Trinity Module (T-type).

    Information processing layer, entrained by D-modules.
    - GAMMA-constrained (|Z| < 0.103)
    - 120° phase signature from 3-node geometry
    - Receives entrainment from D-module master clock

    The 3-node structure requires clamping because:
    - Odd-numbered cycles can accumulate energy
    - 120° = 360°/3 creates resonance that builds
    """

    def __init__(self, hemisphere: str = 'L', name_prefix: str = 'T'):
        super().__init__()

        self.hemisphere = hemisphere.upper()
        self.n_nodes = 3

        # Create 3 T-nodes
        self.nodes = nn.ModuleList([
            DennisNode('T', hemisphere, f"{name_prefix}{i+1}_{hemisphere}")
            for i in range(3)
        ])

        # Coupling weights (triangular topology)
        self.coupling = nn.Parameter(torch.tensor([
            [0.0, 0.1, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ]))

    def forward(self,
                external_input: Optional[torch.Tensor] = None,
                d_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process one timestep.

        Args:
            external_input: Optional external driving signal
            d_phase: Phase from D-module for entrainment

        Returns:
            Z values for all nodes [3]
        """
        z_values = torch.stack([node.Z for node in self.nodes])

        # Weighted neighbor means
        z_neighbors = self.coupling @ z_values

        # Add external input
        if external_input is not None:
            z_neighbors = z_neighbors + external_input * 0.1

        # Update each node with D-module entrainment
        outputs = []
        for i, node in enumerate(self.nodes):
            z_out = node(z_neighbors[i], entrainment_phase=d_phase)
            outputs.append(z_out)

        return torch.stack(outputs)

    def get_state(self) -> Dict:
        return {
            'z': torch.stack([n.Z for n in self.nodes]),
            'phase': torch.stack([n.phase for n in self.nodes]),
            'health': torch.stack([n.health for n in self.nodes]),
        }


class QuadraticModule(nn.Module):
    """
    4-node Quadratic Module (Q-type).

    Complex computation layer, entrained by D-modules.
    - GAMMA-constrained (|Z| < 0.103)
    - Square topology allows both oscillation and computation
    - Higher ceiling pressure than T-modules (needs more constraint)

    The 4-node structure requires clamping because:
    - Cross-diagonal coupling creates feedback loops
    - Even number but complex topology accumulates
    """

    def __init__(self, hemisphere: str = 'L', name_prefix: str = 'Q'):
        super().__init__()

        self.hemisphere = hemisphere.upper()
        self.n_nodes = 4

        # Create 4 Q-nodes
        self.nodes = nn.ModuleList([
            DennisNode('Q', hemisphere, f"{name_prefix}{i+1}_{hemisphere}")
            for i in range(4)
        ])

        # Coupling weights (square topology with diagonals)
        self.coupling = nn.Parameter(torch.tensor([
            [0.0, 0.1, 0.05, 0.1],   # Node 0 connected to 1,3 (adjacent), 2 (diagonal)
            [0.1, 0.0, 0.1, 0.05],   # Node 1 connected to 0,2 (adjacent), 3 (diagonal)
            [0.05, 0.1, 0.0, 0.1],   # Node 2 connected to 1,3 (adjacent), 0 (diagonal)
            [0.1, 0.05, 0.1, 0.0],   # Node 3 connected to 0,2 (adjacent), 1 (diagonal)
        ]))

    def forward(self,
                external_input: Optional[torch.Tensor] = None,
                d_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process one timestep.

        Args:
            external_input: Optional external driving signal
            d_phase: Phase from D-module for entrainment

        Returns:
            Z values for all nodes [4]
        """
        z_values = torch.stack([node.Z for node in self.nodes])

        # Weighted neighbor means
        z_neighbors = self.coupling @ z_values

        # Add external input
        if external_input is not None:
            z_neighbors = z_neighbors + external_input * 0.1

        # Update each node with D-module entrainment
        outputs = []
        for i, node in enumerate(self.nodes):
            z_out = node(z_neighbors[i], entrainment_phase=d_phase)
            outputs.append(z_out)

        return torch.stack(outputs)

    def get_state(self) -> Dict:
        return {
            'z': torch.stack([n.Z for n in self.nodes]),
            'phase': torch.stack([n.phase for n in self.nodes]),
            'health': torch.stack([n.health for n in self.nodes]),
        }


class GenesisHemisphere(nn.Module):
    """
    Complete hemisphere with D, T, and Q modules.

    This is the basic building block for full Genesis networks.
    Contains:
    - 1 DualityModule (2 nodes)
    - 1 TrinityModule (3 nodes)
    - 1 QuadraticModule (4 nodes)
    Total: 9 nodes per hemisphere
    """

    def __init__(self, hemisphere: str = 'L'):
        super().__init__()

        self.hemisphere = hemisphere.upper()

        self.d_module = DualityModule(hemisphere)
        self.t_module = TrinityModule(hemisphere)
        self.q_module = QuadraticModule(hemisphere)

        # Cross-module coupling
        self.dt_coupling = nn.Parameter(torch.tensor(0.05))
        self.tq_coupling = nn.Parameter(torch.tensor(0.05))
        self.dq_coupling = nn.Parameter(torch.tensor(0.02))

    def forward(self, external_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process one timestep through all modules.

        D-module runs first (master clock), then T and Q are entrained.
        """
        # D-module: master clock
        d_out = self.d_module(external_input)
        d_phase = self.d_module.get_mean_phase()

        # T-module: entrained by D
        t_input = d_out.mean() * self.dt_coupling if external_input is None else external_input
        t_out = self.t_module(t_input, d_phase=d_phase)

        # Q-module: entrained by D, receives T
        q_input = t_out.mean() * self.tq_coupling
        q_out = self.q_module(q_input, d_phase=d_phase)

        return {
            'd': d_out,
            't': t_out,
            'q': q_out,
            'd_phase': d_phase,
        }

    def get_all_z(self) -> torch.Tensor:
        """Get all Z values as single tensor [9]."""
        d_state = self.d_module.get_state()
        t_state = self.t_module.get_state()
        q_state = self.q_module.get_state()
        return torch.cat([d_state['z'], t_state['z'], q_state['z']])

    def get_all_phases(self) -> torch.Tensor:
        """Get all phases as single tensor [9]."""
        d_state = self.d_module.get_state()
        t_state = self.t_module.get_state()
        q_state = self.q_module.get_state()
        return torch.cat([d_state['phase'], t_state['phase'], q_state['phase']])
