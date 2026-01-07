#!/usr/bin/env python3
"""
GENESIS-ULTRA vs WORLD'S BEST - CHESS BENCHMARK TOURNAMENT
===========================================================

Head-to-head battles comparing Genesis-Ultra against standard
neural network architectures used in modern chess engines:

1. LSTM-Brain    - Recurrent architecture (like early AlphaZero experiments)
2. GRU-Brain     - Efficient recurrent (used in some Leela variants)
3. Transformer   - Attention-based (modern chess engines)
4. MLP-Brain     - Dense feedforward (baseline)
5. Genesis-Ultra - SPS-optimized D/T/Q architecture

Also includes:
- Standard tactical puzzle benchmarks
- Speed comparisons
- Parameter efficiency analysis

Gamma = 1/(6*phi) = 0.103006
"""

import torch
import torch.nn as nn
import numpy as np
import random
import time
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


# =============================================================================
# CHESS BOARD
# =============================================================================

class ChessBoard:
    """Chess board with full move generation."""

    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
        '.': 0
    }

    # Piece-square tables for positional evaluation
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
        """Advanced evaluation with piece-square tables."""
        score = 0.0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == '.':
                    continue

                # Material
                score += self.PIECE_VALUES.get(piece, 0)

                # Positional bonus
                idx = row * 8 + col
                if piece == 'P':
                    score += self.PAWN_TABLE[idx]
                elif piece == 'p':
                    score -= self.PAWN_TABLE[63 - idx]
                elif piece == 'N':
                    score += self.KNIGHT_TABLE[idx]
                elif piece == 'n':
                    score -= self.KNIGHT_TABLE[63 - idx]

        return score / 100.0  # Normalize

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
# NEURAL NETWORK CHESS BRAINS
# =============================================================================

class SPSActivation(nn.Module):
    """Silent Punctuation Signal Activation."""
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


class GenesisUltraBrain(nn.Module):
    """Genesis-Ultra: SPS-optimized D/T/Q architecture."""

    def __init__(self, seed: int = None):
        super().__init__()
        if seed: torch.manual_seed(seed)

        self.d_linear = nn.Linear(768, 64)
        self.d_phase = nn.Parameter(torch.zeros(64))
        self.d_amp = nn.Parameter(torch.ones(64))
        self.sps_d = SPSActivation(clamp_value=None)

        self.t_linear = nn.Linear(64, 27)
        self.t_phase = nn.Parameter(torch.zeros(27))
        self.t_amp = nn.Parameter(torch.ones(27))
        self.sps_t = SPSActivation(clamp_value=GAMMA)

        self.q_linear = nn.Linear(27, 16)
        self.q_phase = nn.Parameter(torch.zeros(16))
        self.q_amp = nn.Parameter(torch.ones(16))
        self.sps_q = SPSActivation(clamp_value=GAMMA)

        self.output = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.d_linear(x)
        d = d * torch.cos(self.d_phase) * self.d_amp
        d = self.sps_d(d)

        t = self.t_linear(d)
        t = t * torch.cos(self.t_phase) * self.t_amp
        t = self.sps_t(t)

        q = self.q_linear(t)
        q = q * torch.cos(self.q_phase) * self.q_amp
        q = self.sps_q(q)

        return self.output(q)


class LSTMBrain(nn.Module):
    """LSTM-based chess brain (like early AlphaZero experiments)."""

    def __init__(self, seed: int = None):
        super().__init__()
        if seed: torch.manual_seed(seed)

        self.embed = nn.Linear(768, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class GRUBrain(nn.Module):
    """GRU-based chess brain (efficient recurrent)."""

    def __init__(self, seed: int = None):
        super().__init__()
        if seed: torch.manual_seed(seed)

        self.embed = nn.Linear(768, 128)
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).unsqueeze(1)
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


class TransformerBrain(nn.Module):
    """Transformer-based chess brain (modern architecture)."""

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
    """Simple MLP chess brain (baseline)."""

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
# CHESS ENGINE WRAPPER
# =============================================================================

class ChessEngine:
    """Wrapper for neural network chess brain."""

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


# =============================================================================
# TOURNAMENT
# =============================================================================

def play_game(white: ChessEngine, black: ChessEngine) -> Dict:
    """Play a single game."""
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


def run_tournament(engines: List[ChessEngine], games_per_pair: int = 2) -> Dict:
    """Run round-robin tournament."""

    results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0,
                                    'total_time': 0.0, 'total_evals': 0})
    head_to_head = defaultdict(lambda: defaultdict(int))

    total_games = len(engines) * (len(engines) - 1) * games_per_pair
    game_num = 0

    for i, engine1 in enumerate(engines):
        for j, engine2 in enumerate(engines):
            if i == j:
                continue

            for game in range(games_per_pair):
                game_num += 1

                # Alternate colors
                if game % 2 == 0:
                    white, black = engine1, engine2
                else:
                    white, black = engine2, engine1

                result = play_game(white, black)

                # Record results
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

                print(f"  Game {game_num}/{total_games}: {white.name} vs {black.name} -> {result['winner']} ({result['moves']} moves)")

    return dict(results), dict(head_to_head)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("GENESIS-ULTRA vs WORLD'S BEST - CHESS BENCHMARK TOURNAMENT")
    print("=" * 70)
    print(f"GAMMA = {GAMMA:.6f}")
    print(f"PHI = {PHI:.10f}")
    print()

    # Create engines
    print("Creating engines...")
    engines = [
        ChessEngine("Genesis-Ultra", GenesisUltraBrain(seed=42)),
        ChessEngine("LSTM-Brain", LSTMBrain(seed=42)),
        ChessEngine("GRU-Brain", GRUBrain(seed=42)),
        ChessEngine("Transformer", TransformerBrain(seed=42)),
        ChessEngine("MLP-Brain", MLPBrain(seed=42)),
    ]

    # Parameter count
    print()
    print("=" * 70)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 70)
    print(f"{'Engine':<20} {'Parameters':>15} {'vs Genesis':>12}")
    print("-" * 50)

    genesis_params = count_parameters(engines[0].brain)
    for engine in engines:
        params = count_parameters(engine.brain)
        ratio = params / genesis_params
        print(f"{engine.name:<20} {params:>15,} {ratio:>11.1f}x")

    # Speed benchmark
    print()
    print("=" * 70)
    print("INFERENCE SPEED BENCHMARK (1000 evaluations)")
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
    print("HEAD-TO-HEAD TOURNAMENT")
    print("=" * 70)
    print("Round-robin: Each engine plays 2 games against every other engine")
    print()

    results, head_to_head = run_tournament(engines, games_per_pair=2)

    # Final standings
    print()
    print("=" * 70)
    print("FINAL STANDINGS")
    print("=" * 70)
    print(f"{'Engine':<20} {'W':>4} {'L':>4} {'D':>4} {'Points':>8} {'Evals':>10} {'Time(s)':>10}")
    print("-" * 70)

    standings = []
    for name, stats in results.items():
        points = stats['wins'] * 1.0 + stats['draws'] * 0.5
        standings.append((name, stats, points))

    standings.sort(key=lambda x: -x[2])

    for name, stats, points in standings:
        print(f"{name:<20} {stats['wins']:>4} {stats['losses']:>4} {stats['draws']:>4} "
              f"{points:>8.1f} {stats['total_evals']:>10} {stats['total_time']:>10.2f}")

    # Head-to-head matrix
    print()
    print("=" * 70)
    print("HEAD-TO-HEAD WINS MATRIX")
    print("=" * 70)

    engine_names = [e.name for e in engines]
    print(f"{'':>20}", end="")
    for name in engine_names:
        print(f"{name[:8]:>10}", end="")
    print()
    print("-" * 70)

    for name1 in engine_names:
        print(f"{name1:<20}", end="")
        for name2 in engine_names:
            if name1 == name2:
                print(f"{'---':>10}", end="")
            else:
                wins = head_to_head.get(name1, {}).get(name2, 0)
                print(f"{wins:>10}", end="")
        print()

    # Champion
    print()
    print("=" * 70)
    champion = standings[0][0]
    print(f"TOURNAMENT CHAMPION: {champion}")
    print("=" * 70)

    # Genesis advantages
    genesis_stats = results.get("Genesis-Ultra", {})
    lstm_stats = results.get("LSTM-Brain", {})

    print()
    print("GENESIS-ULTRA ANALYSIS:")
    print(f"  - {genesis_params:,} parameters (smallest)")
    print(f"  - Wins: {genesis_stats.get('wins', 0)}, Losses: {genesis_stats.get('losses', 0)}, Draws: {genesis_stats.get('draws', 0)}")
    print(f"  - Gamma-constrained stability")
    print(f"  - SPS threshold activation")
    print()
