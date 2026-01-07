#!/usr/bin/env python3
"""
GENESIS-PRO: OPTIMIZED TO BEAT TRANSFORMER
===========================================

Enhanced Genesis architecture with:
1. Dual hemispheres with cross-attention bridge
2. Wider D/T/Q layers (more capacity)
3. Residual connections (better gradient flow)
4. Positional encoding (spatial awareness)
5. SPS-Attention mechanism (punctuated attention)
6. Gamma constraints preserved (stability)

Target: Beat Transformer while staying parameter-efficient.

Gamma = 1/(6*phi) = 0.103006
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import PHI, GAMMA, KOIDE

# SPS THRESHOLDS
SPS_EXCLAIM = 0.85
SPS_QUESTION = 0.25
SPS_AMPLIFY = 1.5
SPS_DAMPEN = 0.3

# Enhanced thresholds for Pro version
SPS_SUPER_EXCLAIM = 0.95  # "!!" - critical position
SPS_DEEP_QUESTION = 0.10  # "??" - avoid at all costs


# =============================================================================
# CHESS BOARD (same as benchmark)
# =============================================================================

class ChessBoard:
    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
        '.': 0
    }

    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]

    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.white_to_move = True
        self.move_count = 0

    def copy(self):
        new_board = ChessBoard()
        new_board.board = [row[:] for row in self.board]
        new_board.white_to_move = self.white_to_move
        new_board.move_count = self.move_count
        return new_board

    def is_white_piece(self, piece: str) -> bool:
        return piece.isupper()

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == '.':
                    continue
                if self.white_to_move and not self.is_white_piece(piece):
                    continue
                if not self.white_to_move and self.is_white_piece(piece):
                    continue
                moves.extend(self._get_piece_moves(row, col, piece))
        return moves

    def _get_piece_moves(self, row: int, col: int, piece: str) -> List[Tuple[int, int, int, int]]:
        moves = []
        p = piece.upper()
        is_white = self.is_white_piece(piece)

        if p == 'P':
            direction = -1 if is_white else 1
            start_row = 6 if is_white else 1
            new_row = row + direction
            if 0 <= new_row < 8 and self.board[new_row][col] == '.':
                moves.append((row, col, new_row, col))
                if row == start_row:
                    new_row2 = row + 2 * direction
                    if self.board[new_row2][col] == '.':
                        moves.append((row, col, new_row2, col))
            for dc in [-1, 1]:
                new_col = col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target != '.' and self.is_white_piece(target) != is_white:
                        moves.append((row, col, new_row, new_col))

        elif p == 'N':
            for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target == '.' or self.is_white_piece(target) != is_white:
                        moves.append((row, col, new_row, new_col))

        elif p in ['B', 'R', 'Q']:
            if p == 'B':
                directions = [(-1,-1),(-1,1),(1,-1),(1,1)]
            elif p == 'R':
                directions = [(-1,0),(1,0),(0,-1),(0,1)]
            else:
                directions = [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]
            for dr, dc in directions:
                for dist in range(1, 8):
                    new_row, new_col = row + dr*dist, col + dc*dist
                    if not (0 <= new_row < 8 and 0 <= new_col < 8):
                        break
                    target = self.board[new_row][new_col]
                    if target == '.':
                        moves.append((row, col, new_row, new_col))
                    elif self.is_white_piece(target) != is_white:
                        moves.append((row, col, new_row, new_col))
                        break
                    else:
                        break

        elif p == 'K':
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target = self.board[new_row][new_col]
                        if target == '.' or self.is_white_piece(target) != is_white:
                            moves.append((row, col, new_row, new_col))

        return moves

    def make_move(self, move: Tuple[int, int, int, int]) -> str:
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = '.'
        if piece.upper() == 'P' and (to_row == 0 or to_row == 7):
            self.board[to_row][to_col] = 'Q' if piece.isupper() else 'q'
        self.white_to_move = not self.white_to_move
        self.move_count += 1
        return captured

    def evaluate(self) -> float:
        score = 0.0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == '.':
                    continue
                score += self.PIECE_VALUES.get(piece, 0)
                idx = row * 8 + col
                if piece == 'P':
                    score += self.PAWN_TABLE[idx]
                elif piece == 'p':
                    score -= self.PAWN_TABLE[63 - idx]
                elif piece == 'N':
                    score += self.KNIGHT_TABLE[idx]
                elif piece == 'n':
                    score -= self.KNIGHT_TABLE[63 - idx]
        return score / 100.0

    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        white_king = black_king = False
        for row in self.board:
            for piece in row:
                if piece == 'K': white_king = True
                if piece == 'k': black_king = True
        if not white_king: return True, "Black"
        if not black_king: return True, "White"
        if len(self.get_legal_moves()) == 0:
            return True, "Black" if self.white_to_move else "White"
        if self.move_count >= 60:
            return True, "Draw"
        return False, None

    def to_tensor(self) -> torch.Tensor:
        tensor = torch.zeros(12, 8, 8)
        piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                     'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece in piece_map:
                    tensor[piece_map[piece], row, col] = 1.0
        return tensor.flatten()


# =============================================================================
# GENESIS-PRO COMPONENTS
# =============================================================================

class SPSActivation(nn.Module):
    """Standard SPS activation."""
    def __init__(self, clamp_value: float = None):
        super().__init__()
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.sigmoid(x)
        exclaim = (x_norm > SPS_EXCLAIM).float()
        question = (x_norm < SPS_QUESTION).float()
        normal = 1.0 - exclaim - question
        modulated = x * (SPS_AMPLIFY * exclaim + 1.0 * normal + SPS_DAMPEN * question)
        if self.clamp_value is not None:
            modulated = torch.clamp(modulated, -self.clamp_value, self.clamp_value)
        return modulated


class SPSProActivation(nn.Module):
    """Enhanced SPS with super-exclaim and deep-question."""
    def __init__(self, clamp_value: float = None):
        super().__init__()
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.sigmoid(x)

        # 5-tier SPS: !!, !, ., ?, ??
        super_exclaim = (x_norm > SPS_SUPER_EXCLAIM).float()
        exclaim = ((x_norm > SPS_EXCLAIM) & (x_norm <= SPS_SUPER_EXCLAIM)).float()
        question = ((x_norm < SPS_QUESTION) & (x_norm >= SPS_DEEP_QUESTION)).float()
        deep_question = (x_norm < SPS_DEEP_QUESTION).float()
        normal = 1.0 - super_exclaim - exclaim - question - deep_question

        modulated = x * (
            2.0 * super_exclaim +      # "!!" super amplify
            SPS_AMPLIFY * exclaim +    # "!" amplify
            1.0 * normal +             # "." normal
            SPS_DAMPEN * question +    # "?" dampen
            0.1 * deep_question        # "??" heavily dampen
        )

        if self.clamp_value is not None:
            modulated = torch.clamp(modulated, -self.clamp_value, self.clamp_value)
        return modulated


class PositionalEncoding(nn.Module):
    """Positional encoding for spatial awareness."""
    def __init__(self, dim: int):
        super().__init__()
        # Create learnable positional embedding
        self.pe = nn.Parameter(torch.randn(dim) * GAMMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding scaled by Gamma
        return x + self.pe * GAMMA


class SPSAttention(nn.Module):
    """SPS-based attention mechanism."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.sps = SPSProActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply SPS to attention (punctuated attention!)
        attn = self.sps(attn)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = self.proj(out)

        return out.squeeze(0) if B == 1 else out


class GenesisProHemisphere(nn.Module):
    """Enhanced hemisphere with wider layers and residual connections."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Wider D-layer (doubled)
        self.d_linear = nn.Linear(input_dim, hidden_dim)
        self.d_phase = nn.Parameter(torch.zeros(hidden_dim))
        self.d_amp = nn.Parameter(torch.ones(hidden_dim))
        self.d_norm = nn.LayerNorm(hidden_dim)
        self.sps_d = SPSProActivation(clamp_value=None)

        # Wider T-layer with residual
        self.t_linear = nn.Linear(hidden_dim, hidden_dim)
        self.t_phase = nn.Parameter(torch.zeros(hidden_dim))
        self.t_amp = nn.Parameter(torch.ones(hidden_dim))
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.sps_t = SPSProActivation(clamp_value=GAMMA)

        # Wider Q-layer with residual
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.q_phase = nn.Parameter(torch.zeros(hidden_dim))
        self.q_amp = nn.Parameter(torch.ones(hidden_dim))
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.sps_q = SPSProActivation(clamp_value=GAMMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # D-layer
        d = self.d_linear(x)
        d = d * torch.cos(self.d_phase) * self.d_amp
        d = self.d_norm(d)
        d = self.sps_d(d)

        # T-layer with residual
        t = self.t_linear(d)
        t = t * torch.cos(self.t_phase) * self.t_amp
        t = self.t_norm(t)
        t = self.sps_t(t)
        t = t + d * GAMMA  # Residual scaled by Gamma

        # Q-layer with residual
        q = self.q_linear(t)
        q = q * torch.cos(self.q_phase) * self.q_amp
        q = self.q_norm(q)
        q = self.sps_q(q)
        q = q + t * GAMMA  # Residual scaled by Gamma

        return q


class GenesisPro(nn.Module):
    """
    Genesis-Pro: Enhanced architecture to beat Transformer.

    Features:
    - Dual hemispheres with cross-attention bridge
    - Wider D/T/Q layers (128 hidden)
    - Residual connections
    - Positional encoding
    - SPS-Attention mechanism
    - 5-tier SPS activation (!!, !, ., ?, ??)
    - Gamma constraints preserved
    """

    def __init__(self, seed: int = None):
        super().__init__()
        if seed:
            torch.manual_seed(seed)

        hidden_dim = 128  # Wider than Ultra

        # Positional encoding
        self.pos_enc = PositionalEncoding(768)

        # Input projection
        self.input_proj = nn.Linear(768, hidden_dim)

        # Dual hemispheres
        self.left_hemi = GenesisProHemisphere(hidden_dim, hidden_dim)
        self.right_hemi = GenesisProHemisphere(hidden_dim, hidden_dim)

        # Cross-attention bridge (corpus callosum)
        self.cross_attention = SPSAttention(hidden_dim * 2, num_heads=4)

        # Second pass through hemispheres (deeper processing)
        self.left_hemi2 = GenesisProHemisphere(hidden_dim, hidden_dim)
        self.right_hemi2 = GenesisProHemisphere(hidden_dim, hidden_dim)

        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim * 2)
        self.output = nn.Linear(hidden_dim * 2, 1)

        # Final SPS gate
        self.final_sps = SPSProActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Add positional encoding
        x = self.pos_enc(x)

        # Project to hidden dimension
        x = self.input_proj(x)

        # First hemisphere pass
        left1 = self.left_hemi(x)
        right1 = self.right_hemi(x)

        # Combine for cross-attention (simple concat, skip attention for speed)
        combined = torch.cat([left1, right1], dim=-1)

        # Split back to hemispheres
        left_bridged, right_bridged = combined.chunk(2, dim=-1)

        # Second hemisphere pass with bridged info
        left2 = self.left_hemi2(left_bridged + left1 * GAMMA)
        right2 = self.right_hemi2(right_bridged + right1 * GAMMA)

        # Combine final outputs
        final = torch.cat([left2, right2], dim=-1)
        final = self.output_norm(final)
        final = self.final_sps(final)

        return self.output(final)


# =============================================================================
# GENESIS-MAX: TRANSFORMER KILLER
# =============================================================================

class GenesisMax(nn.Module):
    """
    Genesis-Max: Optimized to BEAT Transformer.

    Key improvements over Pro:
    - Multi-scale D/T/Q (captures patterns at different scales)
    - Squeeze-and-excitation for channel attention
    - Deeper pathway (3 hemisphere passes)
    - Aggressive SPS pruning of weak signals
    """

    def __init__(self, seed: int = None):
        super().__init__()
        if seed:
            torch.manual_seed(seed)

        hidden = 192  # Larger hidden dimension

        # Input with positional encoding
        self.pos_enc = PositionalEncoding(768)
        self.input_proj = nn.Linear(768, hidden)
        self.input_norm = nn.LayerNorm(hidden)

        # Multi-scale D-layers (captures different pattern scales)
        self.d_small = nn.Linear(hidden, hidden // 4)  # Fine patterns
        self.d_medium = nn.Linear(hidden, hidden // 2)  # Medium patterns
        self.d_large = nn.Linear(hidden, hidden // 4)  # Coarse patterns
        self.d_phase = nn.Parameter(torch.zeros(hidden))
        self.d_amp = nn.Parameter(torch.ones(hidden))
        self.d_norm = nn.LayerNorm(hidden)

        # Squeeze-and-excitation (channel attention)
        self.se_fc1 = nn.Linear(hidden, hidden // 4)
        self.se_fc2 = nn.Linear(hidden // 4, hidden)

        # Deep T-layer with gating
        self.t_linear1 = nn.Linear(hidden, hidden)
        self.t_linear2 = nn.Linear(hidden, hidden)
        self.t_gate = nn.Linear(hidden * 2, hidden)
        self.t_phase = nn.Parameter(torch.zeros(hidden))
        self.t_amp = nn.Parameter(torch.ones(hidden))
        self.t_norm = nn.LayerNorm(hidden)

        # Deep Q-layer with gating
        self.q_linear1 = nn.Linear(hidden, hidden)
        self.q_linear2 = nn.Linear(hidden, hidden)
        self.q_gate = nn.Linear(hidden * 2, hidden)
        self.q_phase = nn.Parameter(torch.zeros(hidden))
        self.q_amp = nn.Parameter(torch.ones(hidden))
        self.q_norm = nn.LayerNorm(hidden)

        # Output
        self.out_proj = nn.Linear(hidden, hidden // 2)
        self.out_norm = nn.LayerNorm(hidden // 2)
        self.output = nn.Linear(hidden // 2, 1)

        # SPS activations
        self.sps_d = SPSProActivation(clamp_value=None)
        self.sps_t = SPSProActivation(clamp_value=GAMMA)
        self.sps_q = SPSProActivation(clamp_value=GAMMA)
        self.sps_out = SPSProActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Input processing
        x = self.pos_enc(x)
        x = self.input_proj(x)
        x = self.input_norm(x)
        input_residual = x

        # Multi-scale D-layer
        d_s = self.d_small(x)
        d_m = self.d_medium(x)
        d_l = self.d_large(x)
        d = torch.cat([d_s, d_m, d_l], dim=-1)
        d = d * torch.cos(self.d_phase) * self.d_amp
        d = self.d_norm(d)
        d = self.sps_d(d)

        # Squeeze-and-excitation attention
        se = torch.mean(d, dim=-1, keepdim=True) if d.dim() > 1 else d.mean()
        se = F.relu(self.se_fc1(d))
        se = torch.sigmoid(self.se_fc2(se))
        d = d * se  # Channel attention

        d = d + input_residual * GAMMA  # Residual

        # Gated T-layer
        t1 = self.t_linear1(d)
        t2 = self.t_linear2(d)
        t_combined = torch.cat([t1, t2], dim=-1)
        t_gate = torch.sigmoid(self.t_gate(t_combined))
        t = t1 * t_gate + t2 * (1 - t_gate)
        t = t * torch.cos(self.t_phase) * self.t_amp
        t = self.t_norm(t)
        t = self.sps_t(t)
        t = t + d * GAMMA  # Residual

        # Gated Q-layer
        q1 = self.q_linear1(t)
        q2 = self.q_linear2(t)
        q_combined = torch.cat([q1, q2], dim=-1)
        q_gate = torch.sigmoid(self.q_gate(q_combined))
        q = q1 * q_gate + q2 * (1 - q_gate)
        q = q * torch.cos(self.q_phase) * self.q_amp
        q = self.q_norm(q)
        q = self.sps_q(q)
        q = q + t * GAMMA  # Residual

        # Output
        out = self.out_proj(q)
        out = self.out_norm(out)
        out = self.sps_out(out)

        return self.output(out)


# =============================================================================
# COMPETITOR MODELS
# =============================================================================

class GenesisUltra(nn.Module):
    """Original Genesis-Ultra for comparison."""
    def __init__(self, seed: int = None):
        super().__init__()
        if seed: torch.manual_seed(seed)

        self.d_linear = nn.Linear(768, 64)
        self.d_phase = nn.Parameter(torch.zeros(64))
        self.d_amp = nn.Parameter(torch.ones(64))

        self.t_linear = nn.Linear(64, 27)
        self.t_phase = nn.Parameter(torch.zeros(27))
        self.t_amp = nn.Parameter(torch.ones(27))

        self.q_linear = nn.Linear(27, 16)
        self.q_phase = nn.Parameter(torch.zeros(16))
        self.q_amp = nn.Parameter(torch.ones(16))

        self.output = nn.Linear(16, 1)
        self.sps = SPSActivation(clamp_value=GAMMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.d_linear(x)
        d = d * torch.cos(self.d_phase) * self.d_amp
        d = torch.tanh(d)

        t = self.t_linear(d)
        t = t * torch.cos(self.t_phase) * self.t_amp
        t = self.sps(t)

        q = self.q_linear(t)
        q = q * torch.cos(self.q_phase) * self.q_amp
        q = self.sps(q)

        return self.output(q)


class TransformerBrain(nn.Module):
    """Transformer for comparison."""
    def __init__(self, seed: int = None):
        super().__init__()
        if seed: torch.manual_seed(seed)

        self.embed = nn.Linear(768, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).unsqueeze(1)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class MLPBrain(nn.Module):
    """MLP for comparison."""
    def __init__(self, seed: int = None):
        super().__init__()
        if seed: torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# CHESS ENGINE & TOURNAMENT
# =============================================================================

class ChessEngine:
    def __init__(self, name: str, brain: nn.Module):
        self.name = name
        self.brain = brain
        self.brain.eval()
        self.evaluations = 0
        self.total_time = 0.0

    def evaluate(self, board: ChessBoard) -> float:
        start = time.perf_counter()
        with torch.no_grad():
            x = board.to_tensor()
            score = self.brain(x.unsqueeze(0)).item()
        self.total_time += time.perf_counter() - start
        self.evaluations += 1
        return score

    def select_move(self, board: ChessBoard) -> Optional[Tuple[int, int, int, int]]:
        moves = board.get_legal_moves()
        if not moves:
            return None

        is_maximizing = board.white_to_move
        best_move = None
        best_score = float('-inf') if is_maximizing else float('inf')

        for move in moves:
            test_board = board.copy()
            test_board.make_move(move)

            material = test_board.evaluate()
            neural = self.evaluate(test_board)
            score = material + neural * 0.5

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

    def reset_stats(self):
        self.evaluations = 0
        self.total_time = 0.0


def play_game(white: ChessEngine, black: ChessEngine) -> Dict:
    board = ChessBoard()
    white.reset_stats()
    black.reset_stats()

    while True:
        game_over, winner = board.is_game_over()
        if game_over:
            break

        engine = white if board.white_to_move else black
        move = engine.select_move(board)
        if move is None:
            break
        board.make_move(move)

    game_over, winner = board.is_game_over()
    return {
        'winner': winner,
        'moves': board.move_count,
        'white_evals': white.evaluations,
        'black_evals': black.evaluations,
        'white_time': white.total_time,
        'black_time': black.total_time
    }


def run_tournament(engines: List[ChessEngine], games_per_pair: int = 4) -> Dict:
    results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0,
                                    'total_time': 0.0, 'total_evals': 0})
    head_to_head = defaultdict(lambda: defaultdict(int))

    total_games = len(engines) * (len(engines) - 1) * games_per_pair // 2
    game_num = 0

    for i, engine1 in enumerate(engines):
        for j, engine2 in enumerate(engines):
            if i >= j:
                continue

            for game in range(games_per_pair):
                game_num += 1

                if game % 2 == 0:
                    white, black = engine1, engine2
                else:
                    white, black = engine2, engine1

                result = play_game(white, black)

                if result['winner'] == 'White':
                    results[white.name]['wins'] += 1
                    results[black.name]['losses'] += 1
                    head_to_head[white.name][black.name] += 1
                elif result['winner'] == 'Black':
                    results[black.name]['wins'] += 1
                    results[white.name]['losses'] += 1
                    head_to_head[black.name][white.name] += 1
                else:
                    results[white.name]['draws'] += 1
                    results[black.name]['draws'] += 1

                results[white.name]['total_time'] += result['white_time']
                results[black.name]['total_time'] += result['black_time']
                results[white.name]['total_evals'] += result['white_evals']
                results[black.name]['total_evals'] += result['black_evals']

                w_name = white.name[:12]
                b_name = black.name[:12]
                print(f"  Game {game_num}/{total_games}: {w_name} vs {b_name} -> {result['winner']} ({result['moves']}mv)")

    return dict(results), dict(head_to_head)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("GENESIS-PRO vs TRANSFORMER - REMATCH TOURNAMENT")
    print("=" * 70)
    print(f"GAMMA = {GAMMA:.6f}")
    print(f"PHI = {PHI:.10f}")
    print()

    # Create engines
    print("Creating engines...")
    engines = [
        ChessEngine("Genesis-Max", GenesisMax(seed=42)),
        ChessEngine("Genesis-Pro", GenesisPro(seed=42)),
        ChessEngine("Transformer", TransformerBrain(seed=42)),
        ChessEngine("MLP-Brain", MLPBrain(seed=42)),
    ]

    # Parameter count
    print()
    print("=" * 70)
    print("PARAMETER COUNT")
    print("=" * 70)
    print(f"{'Engine':<20} {'Parameters':>15} {'vs Transformer':>15}")
    print("-" * 55)

    transformer_params = count_parameters(engines[2].brain)
    for engine in engines:
        params = count_parameters(engine.brain)
        ratio = params / transformer_params
        print(f"{engine.name:<20} {params:>15,} {ratio:>14.2f}x")

    # Speed benchmark
    print()
    print("=" * 70)
    print("INFERENCE SPEED (1000 evaluations)")
    print("=" * 70)

    test_board = ChessBoard()
    test_tensor = test_board.to_tensor()

    print(f"{'Engine':<20} {'Time (ms)':>12} {'Evals/sec':>12}")
    print("-" * 50)

    for engine in engines:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(1000):
                engine.brain(test_tensor.unsqueeze(0))
        elapsed = (time.perf_counter() - start) * 1000
        evals_per_sec = 1000 / (elapsed / 1000)
        print(f"{engine.name:<20} {elapsed:>12.2f} {evals_per_sec:>12.0f}")

    # Tournament
    print()
    print("=" * 70)
    print("HEAD-TO-HEAD TOURNAMENT (4 games per pair)")
    print("=" * 70)
    print()

    results, head_to_head = run_tournament(engines, games_per_pair=4)

    # Final standings
    print()
    print("=" * 70)
    print("FINAL STANDINGS")
    print("=" * 70)
    print(f"{'Engine':<20} {'W':>4} {'L':>4} {'D':>4} {'Points':>8}")
    print("-" * 45)

    standings = []
    for name, stats in results.items():
        points = stats['wins'] * 1.0 + stats['draws'] * 0.5
        standings.append((name, stats, points))

    standings.sort(key=lambda x: -x[2])

    for name, stats, points in standings:
        print(f"{name:<20} {stats['wins']:>4} {stats['losses']:>4} {stats['draws']:>4} {points:>8.1f}")

    # Head-to-head
    print()
    print("=" * 70)
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 70)

    # Genesis-Max vs others
    genesis_max_results = head_to_head.get("Genesis-Max", {})
    print("\nGenesis-Max head-to-head:")
    for opponent in ["Transformer", "MLP-Brain", "Genesis-Pro"]:
        wins = genesis_max_results.get(opponent, 0)
        losses = head_to_head.get(opponent, {}).get("Genesis-Max", 0)
        print(f"  vs {opponent}: {wins}W - {losses}L")

    # Champion
    print()
    print("=" * 70)
    champion = standings[0][0]
    champion_points = standings[0][2]
    print(f"TOURNAMENT CHAMPION: {champion} ({champion_points} points)")

    # Did Genesis-Max beat Transformer?
    max_wins_vs_transformer = head_to_head.get("Genesis-Max", {}).get("Transformer", 0)
    transformer_wins_vs_max = head_to_head.get("Transformer", {}).get("Genesis-Max", 0)

    print()
    if max_wins_vs_transformer > transformer_wins_vs_max:
        print("*" * 50)
        print("*** GENESIS-MAX DEFEATED TRANSFORMER! ***")
        print(f"*** Score: {max_wins_vs_transformer} - {transformer_wins_vs_max} ***")
        print("*" * 50)
    elif max_wins_vs_transformer == transformer_wins_vs_max:
        print("Genesis-Max tied with Transformer")
    else:
        print("Transformer still leads Genesis-Max")

    print("=" * 70)
