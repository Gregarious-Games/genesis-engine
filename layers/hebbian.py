"""
Phase-Conjugate Hebbian Learning
================================

Hebbian learning with phase-window selection.

Only strengthens connections between nodes whose phase difference
falls within ±(1/PHI) radians (~0.618 rad or ~35°).

This implements the "resonance hunting" behavior from Scaling Forge:
neurons that fire together in phase, wire together.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import numpy as np

from ..core.constants import PHI, GAMMA, DYNAMICS


class PhaseConjugateHebbian(nn.Module):
    """
    Phase-conjugate Hebbian learning layer.

    Updates weights only for node pairs whose phase difference
    is within the golden-ratio window: |Δφ| < 1/PHI radians.

    This is NOT trained via backprop - it's a local learning rule
    that runs alongside the network dynamics.

    Args:
        n_nodes: Number of nodes to track
        learning_rate: Hebbian learning rate (default: GAMMA)
        decay_rate: Weight decay per step (default: GAMMA for slow consolidation)
        phase_window: Phase difference threshold in radians (default: 1/PHI)
    """

    def __init__(self,
                 n_nodes: int,
                 learning_rate: float = None,
                 decay_rate: float = None,
                 phase_window: float = None):
        super().__init__()

        self.n_nodes = n_nodes
        self.lr = learning_rate or DYNAMICS['hebbian_lr']
        self.decay = decay_rate or DYNAMICS['persistent_decay']
        self.window = phase_window or DYNAMICS['hebbian_window']

        # Weight matrix (non-learnable via backprop)
        self.register_buffer('weights', torch.zeros(n_nodes, n_nodes))

        # Delay matrix (for temporal Hebbian)
        self.register_buffer('delays', torch.ones(n_nodes, n_nodes))

        # Update counter
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))

    def forward(self,
                z_values: torch.Tensor,
                phases: torch.Tensor) -> torch.Tensor:
        """
        Apply Hebbian-modulated coupling.

        This is called during the forward pass but does NOT update weights.
        Use update_weights() separately for learning.

        Args:
            z_values: Node activations [n_nodes]
            phases: Node phases [n_nodes]

        Returns:
            hebbian_output: Weighted sum of activations [n_nodes]
        """
        # Hebbian output: each node receives weighted sum from others
        # Positive weights amplify, negative weights suppress
        output = self.weights @ z_values
        return output

    def update_weights(self,
                       z_values: torch.Tensor,
                       phases: torch.Tensor) -> Dict[str, float]:
        """
        Update Hebbian weights based on phase conjugacy.

        This is the local learning rule, separate from backprop.

        Args:
            z_values: Node activations [n_nodes]
            phases: Node phases [n_nodes]

        Returns:
            stats: Dictionary with update statistics
        """
        self.update_count += 1

        # Compute phase differences (all pairs)
        phase_diff = phases.unsqueeze(1) - phases.unsqueeze(0)  # [n, n]

        # Normalize to [-pi, pi]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        # Phase window mask
        in_window = torch.abs(phase_diff) < self.window  # [n, n]

        # Hebbian update: Δw = lr * pre * post * mask
        pre_post = torch.outer(z_values, z_values)  # [n, n]
        delta_w = self.lr * pre_post * in_window.float()

        # Apply update with decay
        self.weights = self.weights * (1 - self.decay) + delta_w

        # Normalize weights to prevent unbounded growth
        max_weight = torch.max(torch.abs(self.weights))
        if max_weight > 1.0:
            self.weights = self.weights / max_weight

        # Update delays based on phase difference
        # Closer phases = shorter delay
        new_delays = 1.0 + torch.abs(phase_diff) / np.pi
        self.delays = 0.99 * self.delays + 0.01 * new_delays

        # Statistics
        n_updates = in_window.sum().item()
        avg_weight = self.weights.mean().item()
        max_w = self.weights.max().item()
        min_w = self.weights.min().item()

        return {
            'n_updates': n_updates,
            'avg_weight': avg_weight,
            'max_weight': max_w,
            'min_weight': min_w,
        }

    def get_strong_connections(self, threshold: float = 0.5) -> list:
        """Get list of strongly connected node pairs."""
        strong = (self.weights > threshold).nonzero(as_tuple=False)
        return [(i.item(), j.item(), self.weights[i, j].item())
                for i, j in strong if i != j]

    def prune_weak_connections(self, threshold: float = -0.5):
        """Remove connections weaker than threshold."""
        self.weights = torch.where(
            self.weights < threshold,
            torch.zeros_like(self.weights),
            self.weights
        )


class HebbianWeightManager:
    """
    Manager for multiple Hebbian layers across a network.

    Handles:
    - Episodic memory (storing snapshots)
    - Dream replay (consolidation)
    - Cross-layer weight analysis
    """

    def __init__(self, max_episodes: int = 50):
        self.max_episodes = max_episodes
        self.episodes = []
        self.layers = {}

    def register_layer(self, name: str, layer: PhaseConjugateHebbian):
        """Register a Hebbian layer for management."""
        self.layers[name] = layer

    def store_episode(self, score: float):
        """Store current weight state as an episode."""
        episode = {
            'score': score,
            'weights': {name: layer.weights.clone()
                        for name, layer in self.layers.items()},
            'timestamp': sum(l.update_count.item() for l in self.layers.values()),
        }

        self.episodes.append(episode)

        # Keep only top episodes
        if len(self.episodes) > self.max_episodes:
            self.episodes.sort(key=lambda e: e['score'], reverse=True)
            self.episodes = self.episodes[:self.max_episodes]

    def replay_best_episode(self, injection_strength: float = 0.1):
        """
        Inject weights from best episode (dream consolidation).

        This is called during "sleep" phases to reinforce good patterns.
        """
        if not self.episodes:
            return

        best = max(self.episodes, key=lambda e: e['score'])

        for name, layer in self.layers.items():
            if name in best['weights']:
                stored = best['weights'][name]
                # Additive injection
                layer.weights = layer.weights + injection_strength * stored

    def get_global_statistics(self) -> Dict:
        """Get statistics across all layers."""
        total_weights = 0
        total_positive = 0
        total_negative = 0

        for name, layer in self.layers.items():
            total_weights += layer.weights.numel()
            total_positive += (layer.weights > 0).sum().item()
            total_negative += (layer.weights < 0).sum().item()

        return {
            'n_layers': len(self.layers),
            'total_weights': total_weights,
            'positive_ratio': total_positive / total_weights if total_weights > 0 else 0,
            'negative_ratio': total_negative / total_weights if total_weights > 0 else 0,
            'n_episodes': len(self.episodes),
        }
