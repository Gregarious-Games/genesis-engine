#!/usr/bin/env python3
"""
GENESIS ENGINE: SPS-OPTIMIZED vs ORIGINAL BENCHMARK
====================================================

Silent Punctuation Signals (SPS) Optimization:
- Vectorized operations (no Python loops)
- Threshold-based semantic processing
- Fused activation functions

SPS Thresholds (from prometheus_chroma.py):
- "!" > 0.85 = Amplify × 1.5 (fresh frontier)
- "." 0.25-0.85 = Normal × 1.0 (standard trail)
- "?" < 0.25 = Dampen × 0.3 (exhausted path)

Gamma = 1/(6*phi) = 0.103006
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple
import gc

# ============================================================================
# GENESIS CONSTANTS
# ============================================================================

PHI = 1.6180339887498949
GAMMA = 1 / (6 * PHI)  # 0.103005664791649
KOIDE = 4 * PHI * GAMMA  # Exactly 2/3
N_EVO = int(120 / PHI)  # 74

# SPS THRESHOLDS (from Starkins Prometheus)
SPS_EXCLAIM = 0.85    # "!" - fresh frontier
SPS_QUESTION = 0.25   # "?" - exhausted path
SPS_AMPLIFY = 1.5     # Amplification factor for "!"
SPS_DAMPEN = 0.3      # Dampening factor for "?"

print("=" * 70)
print("GENESIS ENGINE: SPS-OPTIMIZED vs ORIGINAL")
print("=" * 70)
print(f"PHI = {PHI:.10f}")
print(f"GAMMA = {GAMMA:.10f}")
print(f"SPS Thresholds: ! > {SPS_EXCLAIM}, ? < {SPS_QUESTION}")
print("=" * 70)

# ============================================================================
# ORIGINAL GENESIS (SLOW - Python loops)
# ============================================================================

class DennisNodeOriginal(nn.Module):
    """Original: Per-node phase modulation (SLOW)."""

    def __init__(self, name: str, node_type: str = "D"):
        super().__init__()
        self.name = name
        self.node_type = node_type
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.amplitude = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modulated = x * torch.cos(self.phase) * self.amplitude
        if self.node_type in ["T", "Q"]:
            modulated = torch.clamp(modulated, -GAMMA, GAMMA)
        return modulated


class GenesisOriginal(nn.Module):
    """Original Genesis: 2D + 3T + 4Q with Python loops (SLOW)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Left hemisphere
        self.l_d_layer = nn.Linear(input_dim, 2)
        self.l_d_nodes = nn.ModuleList([DennisNodeOriginal(f"LD{i}", "D") for i in range(2)])
        self.l_t_layer = nn.Linear(2, 3)
        self.l_t_nodes = nn.ModuleList([DennisNodeOriginal(f"LT{i}", "T") for i in range(3)])
        self.l_q_layer = nn.Linear(3, 4)
        self.l_q_nodes = nn.ModuleList([DennisNodeOriginal(f"LQ{i}", "Q") for i in range(4)])
        self.l_out = nn.Linear(4, hidden_dim)

        # Right hemisphere
        self.r_d_layer = nn.Linear(input_dim, 2)
        self.r_d_nodes = nn.ModuleList([DennisNodeOriginal(f"RD{i}", "D") for i in range(2)])
        self.r_t_layer = nn.Linear(2, 3)
        self.r_t_nodes = nn.ModuleList([DennisNodeOriginal(f"RT{i}", "T") for i in range(3)])
        self.r_q_layer = nn.Linear(3, 4)
        self.r_q_nodes = nn.ModuleList([DennisNodeOriginal(f"RQ{i}", "Q") for i in range(4)])
        self.r_out = nn.Linear(4, hidden_dim)

        # Bridge
        self.bridge = nn.Linear(hidden_dim * 2, output_dim)

    def _process_hemisphere(self, x, d_layer, d_nodes, t_layer, t_nodes, q_layer, q_nodes, out_layer):
        # D-layer (SLOW: Python loop)
        d_out = d_layer(x)
        d_out = torch.stack([node(d_out[..., i:i+1]) for i, node in enumerate(d_nodes)], dim=-1)
        d_out = d_out.squeeze(-2)

        # T-layer (SLOW: Python loop)
        t_out = t_layer(d_out)
        t_out = torch.stack([node(t_out[..., i:i+1]) for i, node in enumerate(t_nodes)], dim=-1)
        t_out = t_out.squeeze(-2)

        # Q-layer (SLOW: Python loop)
        q_out = q_layer(t_out)
        q_out = torch.stack([node(q_out[..., i:i+1]) for i, node in enumerate(q_nodes)], dim=-1)
        q_out = q_out.squeeze(-2)

        return out_layer(q_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self._process_hemisphere(x,
            self.l_d_layer, self.l_d_nodes,
            self.l_t_layer, self.l_t_nodes,
            self.l_q_layer, self.l_q_nodes, self.l_out)
        right = self._process_hemisphere(x,
            self.r_d_layer, self.r_d_nodes,
            self.r_t_layer, self.r_t_nodes,
            self.r_q_layer, self.r_q_nodes, self.r_out)
        return self.bridge(torch.cat([left, right], dim=-1))


# ============================================================================
# SPS-OPTIMIZED GENESIS (FAST - Vectorized)
# ============================================================================

class SPSActivation(nn.Module):
    """
    Silent Punctuation Signal Activation.

    Vectorized threshold-based semantic processing:
    - "!" (> 0.85): Amplify × 1.5 - fresh frontier
    - "." (0.25-0.85): Normal × 1.0 - standard trail
    - "?" (< 0.25): Dampen × 0.3 - exhausted path
    """

    def __init__(self, clamp_value: float = None):
        super().__init__()
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to 0-1 range for SPS thresholds
        x_norm = torch.sigmoid(x)

        # SPS threshold masks (vectorized, no loops!)
        exclaim_mask = (x_norm > SPS_EXCLAIM).float()
        question_mask = (x_norm < SPS_QUESTION).float()
        normal_mask = 1.0 - exclaim_mask - question_mask

        # Apply SPS modulation
        modulated = (
            x * SPS_AMPLIFY * exclaim_mask +      # "!" amplify
            x * 1.0 * normal_mask +                # "." normal
            x * SPS_DAMPEN * question_mask         # "?" dampen
        )

        # Apply Gamma clamping if specified (T/Q modules)
        if self.clamp_value is not None:
            modulated = torch.clamp(modulated, -self.clamp_value, self.clamp_value)

        return modulated


class VectorizedDTQLayer(nn.Module):
    """
    Fused D-T-Q layer with SPS activation.

    Single vectorized pass instead of 9 separate node operations.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Fused linear: input -> D(2) -> T(3) -> Q(4) -> output
        # But we keep the structure for interpretability
        self.d_linear = nn.Linear(input_dim, 2)
        self.d_phase = nn.Parameter(torch.zeros(2))
        self.d_amp = nn.Parameter(torch.ones(2))

        self.t_linear = nn.Linear(2, 3)
        self.t_phase = nn.Parameter(torch.zeros(3))
        self.t_amp = nn.Parameter(torch.ones(3))

        self.q_linear = nn.Linear(3, 4)
        self.q_phase = nn.Parameter(torch.zeros(4))
        self.q_amp = nn.Parameter(torch.ones(4))

        self.out_linear = nn.Linear(4, output_dim)

        # SPS activations
        self.sps_d = SPSActivation(clamp_value=None)  # D unbounded
        self.sps_t = SPSActivation(clamp_value=GAMMA)  # T clamped
        self.sps_q = SPSActivation(clamp_value=GAMMA)  # Q clamped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # D-layer: vectorized phase modulation + SPS
        d = self.d_linear(x)
        d = d * torch.cos(self.d_phase) * self.d_amp  # Vectorized!
        d = self.sps_d(d)

        # T-layer: vectorized phase modulation + SPS + Gamma clamp
        t = self.t_linear(d)
        t = t * torch.cos(self.t_phase) * self.t_amp  # Vectorized!
        t = self.sps_t(t)

        # Q-layer: vectorized phase modulation + SPS + Gamma clamp
        q = self.q_linear(t)
        q = q * torch.cos(self.q_phase) * self.q_amp  # Vectorized!
        q = self.sps_q(q)

        return self.out_linear(q)


class GenesisSPS(nn.Module):
    """
    SPS-Optimized Genesis Engine.

    - Vectorized D/T/Q operations (no Python loops)
    - Silent Punctuation Signal activation
    - Fused phase modulation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Two hemispheres with vectorized D-T-Q
        self.left_hemi = VectorizedDTQLayer(input_dim, hidden_dim)
        self.right_hemi = VectorizedDTQLayer(input_dim, hidden_dim)

        # Bridge with SPS activation
        self.bridge = nn.Linear(hidden_dim * 2, output_dim)
        self.bridge_sps = SPSActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left_hemi(x)
        right = self.right_hemi(x)
        combined = torch.cat([left, right], dim=-1)
        return self.bridge_sps(self.bridge(combined))


# ============================================================================
# ULTRA-OPTIMIZED: FULLY FUSED VERSION
# ============================================================================

class GenesisUltra(nn.Module):
    """
    Ultra-optimized Genesis with fully fused operations.

    Single matrix multiplication path with SPS at output only.
    Maximum CUDA efficiency.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Fused pathway: input -> 9 nodes -> hidden -> output
        # D(2) + T(3) + Q(4) = 9 internal nodes per hemisphere = 18 total

        self.left_fused = nn.Sequential(
            nn.Linear(input_dim, 9),  # D+T+Q fused
            nn.Tanh(),  # Bounded activation
        )
        self.right_fused = nn.Sequential(
            nn.Linear(input_dim, 9),
            nn.Tanh(),
        )

        # Gamma-constrained output
        self.out = nn.Linear(18, hidden_dim)
        self.final = nn.Linear(hidden_dim, output_dim)

        # Learnable SPS thresholds
        self.exclaim_thresh = nn.Parameter(torch.tensor(SPS_EXCLAIM))
        self.question_thresh = nn.Parameter(torch.tensor(SPS_QUESTION))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left_fused(x)
        right = self.right_fused(x)

        combined = torch.cat([left, right], dim=-1)
        h = self.out(combined)

        # SPS at output only (fastest)
        h_norm = torch.sigmoid(h)
        exclaim = (h_norm > self.exclaim_thresh).float()
        question = (h_norm < self.question_thresh).float()
        normal = 1.0 - exclaim - question

        h = h * (SPS_AMPLIFY * exclaim + 1.0 * normal + SPS_DAMPEN * question)
        h = torch.clamp(h, -GAMMA, GAMMA)  # Gamma constraint

        return self.final(h)


# ============================================================================
# BENCHMARK COMPARISONS
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(model: nn.Module, input_shape: Tuple[int, ...],
                        num_runs: int = 1000, warmup: int = 100) -> Dict:
    model.eval()
    x = torch.randn(*input_shape)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - start)

    times = np.array(times) * 1000
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'throughput': 1000 / np.mean(times)
    }


def benchmark_training(model: nn.Module, input_shape: Tuple[int, ...],
                       output_dim: int, num_steps: int = 100) -> Dict:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    x = torch.randn(*input_shape)
    y = torch.randn(input_shape[0], output_dim)

    for _ in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    return {
        'time_per_step_ms': (elapsed / num_steps) * 1000,
        'steps_per_second': num_steps / elapsed
    }


def run_benchmark():
    INPUT_DIM = 64
    HIDDEN_DIM = 64
    OUTPUT_DIM = 10
    BATCH_SIZE = 32

    input_shape = (BATCH_SIZE, INPUT_DIM)

    print(f"\nConfiguration:")
    print(f"  Input/Hidden/Output: {INPUT_DIM}/{HIDDEN_DIM}/{OUTPUT_DIM}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()

    models = {
        "Genesis-Original": GenesisOriginal(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        "Genesis-SPS": GenesisSPS(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        "Genesis-Ultra": GenesisUltra(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        "LSTM": LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        "GRU": GRUModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
    }

    results = {}

    # Parameter count
    print("=" * 70)
    print("PARAMETER COUNT")
    print("=" * 70)
    print(f"{'Model':<20} {'Parameters':>12}")
    print("-" * 35)

    for name, model in models.items():
        params = count_parameters(model)
        results[name] = {"params": params}
        print(f"{name:<20} {params:>12,}")

    # Inference benchmark
    print()
    print("=" * 70)
    print("INFERENCE BENCHMARK (1000 runs)")
    print("=" * 70)
    print(f"{'Model':<20} {'Mean (ms)':>10} {'Throughput':>15}")
    print("-" * 50)

    for name, model in models.items():
        gc.collect()
        bench = benchmark_inference(model, input_shape)
        results[name].update(bench)
        print(f"{name:<20} {bench['mean_ms']:>10.3f} {bench['throughput']:>13.0f}/s")

    # Training benchmark
    print()
    print("=" * 70)
    print("TRAINING BENCHMARK (100 steps)")
    print("=" * 70)
    print(f"{'Model':<20} {'Time/step':>12} {'Steps/sec':>12}")
    print("-" * 50)

    for name in models:
        gc.collect()
        # Recreate model
        if name == "Genesis-Original":
            model = GenesisOriginal(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        elif name == "Genesis-SPS":
            model = GenesisSPS(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        elif name == "Genesis-Ultra":
            model = GenesisUltra(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        elif name == "LSTM":
            model = LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        elif name == "GRU":
            model = GRUModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

        train_bench = benchmark_training(model, input_shape, OUTPUT_DIM)
        results[name]["train_time"] = train_bench["time_per_step_ms"]
        results[name]["train_speed"] = train_bench["steps_per_second"]
        print(f"{name:<20} {train_bench['time_per_step_ms']:>10.3f}ms {train_bench['steps_per_second']:>10.1f}/s")

    # Speedup summary
    print()
    print("=" * 70)
    print("SPS OPTIMIZATION SPEEDUP")
    print("=" * 70)

    orig = results["Genesis-Original"]
    sps = results["Genesis-SPS"]
    ultra = results["Genesis-Ultra"]

    print(f"\nGenesis-SPS vs Genesis-Original:")
    print(f"  Inference: {orig['mean_ms']/sps['mean_ms']:.2f}x faster")
    print(f"  Training:  {orig['train_time']/sps['train_time']:.2f}x faster")

    print(f"\nGenesis-Ultra vs Genesis-Original:")
    print(f"  Inference: {orig['mean_ms']/ultra['mean_ms']:.2f}x faster")
    print(f"  Training:  {orig['train_time']/ultra['train_time']:.2f}x faster")

    print(f"\nGenesis-Ultra vs LSTM:")
    lstm = results["LSTM"]
    print(f"  Parameters: {lstm['params']/ultra['params']:.1f}x fewer in Genesis")
    print(f"  Inference:  {lstm['mean_ms']/ultra['mean_ms']:.2f}x {'faster' if ultra['mean_ms'] < lstm['mean_ms'] else 'slower'}")

    print()
    print("=" * 70)
    print("SPS TECHNOLOGY APPLIED")
    print("=" * 70)
    print("""
Silent Punctuation Signals (SPS):
  "!" > 0.85  ->  Amplify x 1.5  (fresh frontier)
  "." 0.25-0.85  ->  Normal x 1.0  (standard trail)
  "?" < 0.25  ->  Dampen x 0.3  (exhausted path)

Optimizations applied:
  1. Vectorized phase modulation (no Python loops)
  2. Fused D/T/Q operations
  3. Threshold-based SPS activation
  4. Gamma clamping preserved for stability
""")

    return results


if __name__ == "__main__":
    results = run_benchmark()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
