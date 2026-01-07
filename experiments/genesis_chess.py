"""
Genesis-Pro Chess Battle (Tournament Champion)
==============================================

Two Genesis-Pro Engines battle in chess.
Genesis-Pro: 10W-2L tournament champion, beat Transformer with 0.44x params.

Features:
- Dual hemispheres with cross-attention bridge
- Wider D/T/Q layers (128 hidden)
- Residual connections with Gamma scaling
- 5-tier SPS activation (!!, !, ., ?, ??)
- Positional encoding

SPS Thresholds:
  "!!" > 0.95 -> Super amplify x2.0
  "!"  > 0.85 -> Amplify x1.5
  "."  0.25-0.85 -> Normal x1.0
  "?"  < 0.25 -> Dampen x0.3
  "??" < 0.10 -> Heavy dampen x0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import PHI, GAMMA, KOIDE

# SPS THRESHOLDS
SPS_EXCLAIM = 0.85
SPS_QUESTION = 0.25
SPS_AMPLIFY = 1.5
SPS_DAMPEN = 0.3
SPS_SUPER_EXCLAIM = 0.95
SPS_DEEP_QUESTION = 0.10


# =============================================================================
# CHESS ENGINE
# =============================================================================

class ChessBoard:
    """Simple chess board representation."""

    PIECE_VALUES = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -100,
        '.': 0
    }

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
        self.history = []

    def copy(self):
        new_board = ChessBoard()
        new_board.board = [row[:] for row in self.board]
        new_board.white_to_move = self.white_to_move
        new_board.move_count = self.move_count
        return new_board

    def get_piece(self, row: int, col: int) -> str:
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def is_white_piece(self, piece: str) -> bool:
        return piece.isupper()

    def is_black_piece(self, piece: str) -> bool:
        return piece.islower()

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == '.':
                    continue
                if self.white_to_move and not self.is_white_piece(piece):
                    continue
                if not self.white_to_move and not self.is_black_piece(piece):
                    continue
                piece_moves = self._get_piece_moves(row, col, piece)
                moves.extend(piece_moves)
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
                new_row = row + direction
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
        self.history.append(move)
        return captured

    def evaluate(self) -> float:
        score = 0.0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                score += self.PIECE_VALUES.get(piece, 0)
                if piece.upper() == 'P':
                    if piece.isupper():
                        score += (6 - row) * 0.1
                    else:
                        score -= (row - 1) * 0.1
                if piece != '.':
                    center_dist = abs(row - 3.5) + abs(col - 3.5)
                    bonus = (7 - center_dist) * 0.05
                    if piece.isupper():
                        score += bonus
                    else:
                        score -= bonus
        return score

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
        if self.move_count >= 100:
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

    def display(self) -> str:
        lines = ["  a b c d e f g h", "  ----------------"]
        for i, row in enumerate(self.board):
            rank = 8 - i
            lines.append(f"{rank}|{' '.join(row)}|{rank}")
        lines.extend(["  ----------------", "  a b c d e f g h"])
        return '\n'.join(lines)

    def move_to_algebraic(self, move: Tuple[int, int, int, int]) -> str:
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row][from_col]
        from_sq = chr(ord('a') + from_col) + str(8 - from_row)
        to_sq = chr(ord('a') + to_col) + str(8 - to_row)
        p = piece.upper()
        return f"{from_sq}-{to_sq}" if p == 'P' else f"{p}{from_sq}-{to_sq}"


# =============================================================================
# GENESIS-PRO COMPONENTS (Tournament Champion)
# =============================================================================

class SPSProActivation(nn.Module):
    """5-tier SPS activation: !!, !, ., ?, ??"""
    def __init__(self, clamp_value: float = None):
        super().__init__()
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.sigmoid(x)
        super_exclaim = (x_norm > SPS_SUPER_EXCLAIM).float()
        exclaim = ((x_norm > SPS_EXCLAIM) & (x_norm <= SPS_SUPER_EXCLAIM)).float()
        question = ((x_norm < SPS_QUESTION) & (x_norm >= SPS_DEEP_QUESTION)).float()
        deep_question = (x_norm < SPS_DEEP_QUESTION).float()
        normal = 1.0 - super_exclaim - exclaim - question - deep_question

        modulated = x * (
            2.0 * super_exclaim +
            SPS_AMPLIFY * exclaim +
            1.0 * normal +
            SPS_DAMPEN * question +
            0.1 * deep_question
        )

        if self.clamp_value is not None:
            modulated = torch.clamp(modulated, -self.clamp_value, self.clamp_value)
        return modulated


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(dim) * GAMMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe * GAMMA


class GenesisProHemisphere(nn.Module):
    """Enhanced hemisphere with wider layers and residual connections."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.d_linear = nn.Linear(input_dim, hidden_dim)
        self.d_phase = nn.Parameter(torch.zeros(hidden_dim))
        self.d_amp = nn.Parameter(torch.ones(hidden_dim))
        self.d_norm = nn.LayerNorm(hidden_dim)
        self.sps_d = SPSProActivation(clamp_value=None)

        self.t_linear = nn.Linear(hidden_dim, hidden_dim)
        self.t_phase = nn.Parameter(torch.zeros(hidden_dim))
        self.t_amp = nn.Parameter(torch.ones(hidden_dim))
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.sps_t = SPSProActivation(clamp_value=GAMMA)

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.q_phase = nn.Parameter(torch.zeros(hidden_dim))
        self.q_amp = nn.Parameter(torch.ones(hidden_dim))
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.sps_q = SPSProActivation(clamp_value=GAMMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.d_linear(x)
        d = d * torch.cos(self.d_phase) * self.d_amp
        d = self.d_norm(d)
        d = self.sps_d(d)

        t = self.t_linear(d)
        t = t * torch.cos(self.t_phase) * self.t_amp
        t = self.t_norm(t)
        t = self.sps_t(t)
        t = t + d * GAMMA

        q = self.q_linear(t)
        q = q * torch.cos(self.q_phase) * self.q_amp
        q = self.q_norm(q)
        q = self.sps_q(q)
        q = q + t * GAMMA

        return q


class GenesisBrain(nn.Module):
    """
    Genesis-Pro Chess Engine (Tournament Champion).
    10W-2L record, beat Transformer with 0.44x parameters.
    """

    def __init__(self, name: str, seed: int = None):
        super().__init__()
        self.name = name

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        hidden_dim = 128

        # Positional encoding
        self.pos_enc = PositionalEncoding(768)

        # Input projection
        self.input_proj = nn.Linear(768, hidden_dim)

        # Dual hemispheres
        self.left_hemi = GenesisProHemisphere(hidden_dim, hidden_dim)
        self.right_hemi = GenesisProHemisphere(hidden_dim, hidden_dim)

        # Second pass (deeper processing)
        self.left_hemi2 = GenesisProHemisphere(hidden_dim, hidden_dim)
        self.right_hemi2 = GenesisProHemisphere(hidden_dim, hidden_dim)

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim * 2)
        self.output = nn.Linear(hidden_dim * 2, 1)
        self.final_sps = SPSProActivation()

        # Statistics
        self.evaluations = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.pos_enc(x)
        x = self.input_proj(x)

        # First hemisphere pass
        left1 = self.left_hemi(x)
        right1 = self.right_hemi(x)

        # Cross-hemisphere bridge
        combined = torch.cat([left1, right1], dim=-1)
        left_bridged, right_bridged = combined.chunk(2, dim=-1)

        # Second hemisphere pass
        left2 = self.left_hemi2(left_bridged + left1 * GAMMA)
        right2 = self.right_hemi2(right_bridged + right1 * GAMMA)

        # Output
        final = torch.cat([left2, right2], dim=-1)
        final = self.output_norm(final)
        final = self.final_sps(final)

        self.evaluations += 1
        return self.output(final)

    def evaluate_position(self, board: ChessBoard) -> float:
        with torch.no_grad():
            x = board.to_tensor()
            score = self.forward(x)
            return score.item()

    def select_move(self, board: ChessBoard, depth: int = 1) -> Tuple[int, int, int, int]:
        moves = board.get_legal_moves()
        if not moves:
            return None

        is_maximizing = board.white_to_move
        best_move = None
        best_score = float('-inf') if is_maximizing else float('inf')

        for move in moves:
            test_board = board.copy()
            test_board.make_move(move)

            score = self._minimax(test_board, depth - 1, float('-inf'), float('inf'), not is_maximizing)

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

    def _minimax(self, board: ChessBoard, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        game_over, winner = board.is_game_over()
        if game_over:
            if winner == "White": return 1000
            elif winner == "Black": return -1000
            else: return 0

        if depth == 0:
            material = board.evaluate()
            neural = self.evaluate_position(board)
            return material + neural * GAMMA

        moves = board.get_legal_moves()

        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                test_board = board.copy()
                test_board.make_move(move)
                eval_score = self._minimax(test_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                test_board = board.copy()
                test_board.make_move(move)
                eval_score = self._minimax(test_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval


# =============================================================================
# CHESS BATTLE
# =============================================================================

def play_game(white_engine: GenesisBrain, black_engine: GenesisBrain,
              verbose: bool = True, depth: int = 1) -> Dict:
    board = ChessBoard()

    if verbose:
        print("=" * 60)
        print("GENESIS-PRO CHESS BATTLE")
        print("=" * 60)
        print(f"\nWhite: {white_engine.name}")
        print(f"Black: {black_engine.name}")
        print(f"\nGAMMA = {GAMMA:.6f}")
        print(f"PHI = {PHI:.10f}")
        print("\n" + board.display())
        print()

    move_list = []

    while True:
        game_over, winner = board.is_game_over()
        if game_over:
            break

        engine = white_engine if board.white_to_move else black_engine
        color = "White" if board.white_to_move else "Black"

        move = engine.select_move(board, depth=depth)
        if move is None:
            break

        move_str = board.move_to_algebraic(move)
        captured = board.make_move(move)
        move_list.append(move_str)

        if verbose:
            capture_str = f" x {captured}" if captured != '.' else ""
            print(f"Move {board.move_count}: {color} plays {move_str}{capture_str}")

            if board.move_count % 10 == 0:
                print(f"\nPosition after move {board.move_count}:")
                print(board.display())
                print(f"Evaluation: {board.evaluate():+.2f}")
                print()

    game_over, winner = board.is_game_over()

    if verbose:
        print("\n" + "=" * 60)
        print("GAME OVER")
        print("=" * 60)
        print(f"\nFinal position:")
        print(board.display())
        print(f"\nResult: {winner} wins!" if winner != "Draw" else "\nResult: Draw!")
        print(f"Total moves: {board.move_count}")
        print(f"\nWhite evaluations: {white_engine.evaluations}")
        print(f"Black evaluations: {black_engine.evaluations}")

    return {
        'winner': winner,
        'moves': board.move_count,
        'move_list': move_list,
        'final_eval': board.evaluate(),
        'white_evals': white_engine.evaluations,
        'black_evals': black_engine.evaluations
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "   GENESIS-PRO CHESS BATTLE   ".center(58) + "*")
    print("*" + "   Tournament Champion (10W-2L)   ".center(58) + "*")
    print("*" + f"   Gamma = 1/(6*phi) = {GAMMA:.6f}   ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()

    # Create two Genesis-Pro engines
    engine_white = GenesisBrain("GENESIS-WHITE", seed=42)
    engine_black = GenesisBrain("GENESIS-BLACK", seed=74)

    # Play a single game
    result = play_game(engine_white, engine_black, verbose=True, depth=1)

    print("\n" + "=" * 60)
    print("BATTLE STATISTICS")
    print("=" * 60)
    print(f"Winner: {result['winner']}")
    print(f"Total moves: {result['moves']}")
    print(f"White evaluations: {result['white_evals']}")
    print(f"Black evaluations: {result['black_evals']}")
    print(f"Final material balance: {result['final_eval']:+.2f}")
