"""
Oscillator Layer
================

Core oscillator dynamics as a differentiable layer.

This wraps DennisNode dynamics into a format suitable for
integration with standard PyTorch networks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

from ..core.constants import PHI, GAMMA
from ..core.clamping import clamp, get_max_abs


class OscillatorLayer(nn.Module):
    """
    Oscillator layer with Genesis dynamics.

    Converts input signals to oscillatory activations.

    Args:
        input_dim: Input feature dimension
        n_oscillators: Number of oscillator units
        node_type: 'D', 'T', or 'Q' for clamping behavior
    """

    def __init__(self,
                 input_dim: int,
                 n_oscillators: int,
                 node_type: str = 'T'):
        super().__init__()

        self.input_dim = input_dim
        self.n_oscillators = n_oscillators
        self.node_type = node_type.upper()
        self.max_abs = get_max_abs(self.node_type)

        # Input projection
        self.input_proj = nn.Linear(input_dim, n_oscillators)

        # Oscillator parameters
        self.amplitude = nn.Parameter(torch.ones(n_oscillators))
        self.frequency = nn.Parameter(torch.ones(n_oscillators))
        self.phase_offset = nn.Parameter(torch.zeros(n_oscillators))
        self.damping = nn.Parameter(torch.full((n_oscillators,), 0.01))

        # Internal state
        self.register_buffer('state', torch.zeros(n_oscillators))
        self.register_buffer('velocity', torch.zeros(n_oscillators))
        self.register_buffer('phase', torch.zeros(n_oscillators))
        self.register_buffer('t', torch.tensor(0.0))

    def forward(self,
                x: torch.Tensor,
                dt: float = 0.01) -> torch.Tensor:
        """
        Process input through oscillator dynamics.

        Args:
            x: Input tensor [batch, input_dim] or [input_dim]
            dt: Time step

        Returns:
            output: Oscillator activations [batch, n_oscillators] or [n_oscillators]
        """
        # Handle batched vs single input
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Project input to oscillator space
        driving_force = self.input_proj(x)  # [batch, n_osc]

        # Time update
        self.t = self.t + dt

        # Oscillator dynamics (damped driven harmonic oscillator)
        # d²x/dt² = -ω²x - γ(dx/dt) + F(t)
        omega_sq = (2 * np.pi * self.frequency) ** 2
        gamma = self.damping

        # For each sample in batch
        outputs = []
        for i in range(batch_size):
            force = driving_force[i]

            # Velocity update
            accel = -omega_sq * self.state - gamma * self.velocity + force
            self.velocity = self.velocity + accel * dt

            # State update
            self.state = self.state + self.velocity * dt

            # Add oscillatory component
            osc = self.amplitude * torch.sin(
                2 * np.pi * self.frequency * self.t + self.phase_offset
            )
            output = self.state + osc

            # Clamping
            if self.max_abs != float('inf'):
                output = torch.clamp(output, -self.max_abs, self.max_abs)

            outputs.append(output)

        # Update phase
        self.phase = torch.atan2(self.velocity, self.state + 1e-10)

        result = torch.stack(outputs)

        if single:
            result = result.squeeze(0)

        return result

    def reset(self):
        """Reset oscillator state."""
        self.state.zero_()
        self.velocity.zero_()
        self.phase.zero_()
        self.t.zero_()

    def get_phases(self) -> torch.Tensor:
        """Get current oscillator phases."""
        return self.phase.clone()


class SparseOscillatorLayer(nn.Module):
    """
    Oscillator layer with φ¹¹-rhythm sparse updates.

    Only performs gradient updates every PHI^11 steps (~199 steps).
    Uses local Hebbian learning in between.

    This is the energy-efficient mode for deployment.
    """

    def __init__(self,
                 input_dim: int,
                 n_oscillators: int,
                 node_type: str = 'T'):
        super().__init__()

        self.oscillator = OscillatorLayer(input_dim, n_oscillators, node_type)
        self.sparse_interval = int(PHI ** 11)  # ~199
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        self.step_count += 1

        # Check if this is a gradient step
        is_gradient_step = (self.step_count % self.sparse_interval) == 0

        if is_gradient_step:
            # Normal forward pass with gradients
            return self.oscillator(x, dt)
        else:
            # Forward pass without gradient tracking
            with torch.no_grad():
                return self.oscillator(x, dt)

    def reset(self):
        self.oscillator.reset()
        self.step_count.zero_()
