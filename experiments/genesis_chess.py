"""
Genesis Engine Chess Battle
============================

Two Genesis Engines (Gamma-constrained oscillator networks) battle in chess.
Each engine uses its D/T/Q module dynamics to evaluate positions and select moves.

This is a demonstration of the Genesis architecture applied to game playing.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import PHI, GAMMA, KOIDE


# =============================================================================
# CHESS ENGINE
# =============================================================================

class ChessBoard:
    """Simple chess board representation."""

    # Piece values for evaluation
    PIECE_VALUES = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -100,
        '.': 0
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Set up initial position."""
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
        """Create a copy of the board."""
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
        """Get all legal moves for current player. Returns list of (from_row, from_col, to_row, to_col)."""
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
        """Get moves for a specific piece."""
        moves = []
        p = piece.upper()
        is_white = self.is_white_piece(piece)

        if p == 'P':
            # Pawn moves
            direction = -1 if is_white else 1
            start_row = 6 if is_white else 1

            # Forward move
            new_row = row + direction
            if 0 <= new_row < 8 and self.board[new_row][col] == '.':
                moves.append((row, col, new_row, col))
                # Double move from start
                if row == start_row:
                    new_row2 = row + 2 * direction
                    if self.board[new_row2][col] == '.':
                        moves.append((row, col, new_row2, col))

            # Captures
            for dc in [-1, 1]:
                new_col = col + dc
                new_row = row + direction
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target != '.' and self.is_white_piece(target) != is_white:
                        moves.append((row, col, new_row, new_col))

        elif p == 'N':
            # Knight moves
            for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target == '.' or self.is_white_piece(target) != is_white:
                        moves.append((row, col, new_row, new_col))

        elif p == 'B':
            # Bishop moves
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
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

        elif p == 'R':
            # Rook moves
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
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

        elif p == 'Q':
            # Queen = Bishop + Rook
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]:
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
            # King moves
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
        """Make a move. Returns captured piece or '.'"""
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]

        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = '.'

        # Pawn promotion
        if piece.upper() == 'P' and (to_row == 0 or to_row == 7):
            self.board[to_row][to_col] = 'Q' if piece.isupper() else 'q'

        self.white_to_move = not self.white_to_move
        self.move_count += 1
        self.history.append(move)

        return captured

    def evaluate(self) -> float:
        """Evaluate position. Positive = white advantage."""
        score = 0.0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                score += self.PIECE_VALUES.get(piece, 0)

                # Positional bonuses
                if piece.upper() == 'P':
                    # Pawns more valuable as they advance
                    if piece.isupper():
                        score += (6 - row) * 0.1
                    else:
                        score -= (row - 1) * 0.1

                # Center control
                if piece != '.':
                    center_dist = abs(row - 3.5) + abs(col - 3.5)
                    bonus = (7 - center_dist) * 0.05
                    if piece.isupper():
                        score += bonus
                    else:
                        score -= bonus

        return score

    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """Check if game is over. Returns (is_over, winner)."""
        # Check for kings
        white_king = False
        black_king = False
        for row in self.board:
            for piece in row:
                if piece == 'K':
                    white_king = True
                if piece == 'k':
                    black_king = True

        if not white_king:
            return True, "Black"
        if not black_king:
            return True, "White"

        # Check for no legal moves (simplified - doesn't check for stalemate vs checkmate)
        if len(self.get_legal_moves()) == 0:
            return True, "Black" if self.white_to_move else "White"

        # Draw by move limit
        if self.move_count >= 100:
            return True, "Draw"

        return False, None

    def to_tensor(self) -> torch.Tensor:
        """Convert board to tensor for neural network input."""
        # 12 channels: 6 piece types x 2 colors
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
        """Return string representation of board."""
        lines = []
        lines.append("  a b c d e f g h")
        lines.append("  ----------------")
        for i, row in enumerate(self.board):
            rank = 8 - i
            lines.append(f"{rank}|{' '.join(row)}|{rank}")
        lines.append("  ----------------")
        lines.append("  a b c d e f g h")
        return '\n'.join(lines)

    def move_to_algebraic(self, move: Tuple[int, int, int, int]) -> str:
        """Convert move to algebraic notation."""
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row][from_col]

        from_sq = chr(ord('a') + from_col) + str(8 - from_row)
        to_sq = chr(ord('a') + to_col) + str(8 - to_row)

        p = piece.upper()
        if p == 'P':
            return f"{from_sq}-{to_sq}"
        else:
            return f"{p}{from_sq}-{to_sq}"


# =============================================================================
# GENESIS CHESS ENGINE
# =============================================================================

class GenesisBrain(nn.Module):
    """
    Genesis Engine brain for chess.
    Uses D/T/Q module architecture with Gamma constraints.
    """

    def __init__(self, name: str, seed: int = None):
        super().__init__()
        self.name = name

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Input: 768 (12 channels x 64 squares)
        # D-module: unbounded, self-regulating
        self.d_layer = nn.Linear(768, 64)

        # T-module: Gamma-constrained (3-node structure)
        self.t_layer = nn.Linear(64, 27)  # 3^3 = 27

        # Q-module: Gamma-constrained (4-node structure)
        self.q_layer = nn.Linear(27, 16)  # 4^2 = 16

        # Output layer
        self.output = nn.Linear(16, 1)

        # Phase states for oscillator dynamics
        self.d_phase = torch.zeros(64)
        self.t_phase = torch.zeros(27)
        self.q_phase = torch.zeros(16)

        # Oscillator frequency based on golden ratio
        self.omega = 2 * np.pi / PHI

        # Statistics
        self.evaluations = 0
        self.total_time = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate position using Genesis dynamics."""
        # D-module: unbounded, uses tanh for self-regulation
        d_out = torch.tanh(self.d_layer(x))

        # Update D-phase (Kuramoto-style)
        self.d_phase = self.d_phase + self.omega * 0.1 + 0.1 * torch.sin(d_out)

        # T-module: Gamma-constrained
        t_out = self.t_layer(d_out)
        t_out = torch.clamp(t_out, -GAMMA, GAMMA)  # Key constraint!

        # Update T-phase with 120-degree attractor
        self.t_phase = self.t_phase + self.omega * 0.1

        # Q-module: Gamma-constrained
        q_out = self.q_layer(t_out)
        q_out = torch.clamp(q_out, -GAMMA, GAMMA)  # Key constraint!

        # Update Q-phase
        self.q_phase = self.q_phase + self.omega * 0.1

        # Final evaluation
        score = self.output(q_out)

        self.evaluations += 1
        return score

    def evaluate_position(self, board: ChessBoard) -> float:
        """Evaluate a chess position."""
        with torch.no_grad():
            x = board.to_tensor()
            score = self.forward(x)
            return score.item()

    def select_move(self, board: ChessBoard, depth: int = 2) -> Tuple[int, int, int, int]:
        """Select best move using minimax with Genesis evaluation."""
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

            # Add some Genesis-style noise based on phase
            phase_noise = 0.01 * np.sin(self.d_phase.mean().item())
            score += phase_noise

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
        """Minimax with alpha-beta pruning."""
        game_over, winner = board.is_game_over()
        if game_over:
            if winner == "White":
                return 1000
            elif winner == "Black":
                return -1000
            else:
                return 0

        if depth == 0:
            # Combine material evaluation with Genesis neural evaluation
            material = board.evaluate()
            neural = self.evaluate_position(board)
            return material + neural * GAMMA  # Weight neural eval by Gamma

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
              verbose: bool = True, depth: int = 2) -> Dict:
    """Play a game between two Genesis engines."""

    board = ChessBoard()

    if verbose:
        print("=" * 60)
        print("GENESIS ENGINE CHESS BATTLE")
        print("=" * 60)
        print(f"\nWhite: {white_engine.name}")
        print(f"Black: {black_engine.name}")
        print(f"\nGAMMA = {GAMMA:.6f}")
        print(f"PHI = {PHI:.10f}")
        print(f"KOIDE = {KOIDE:.10f}")
        print("\n" + board.display())
        print()

    move_list = []

    while True:
        game_over, winner = board.is_game_over()
        if game_over:
            break

        # Select engine
        if board.white_to_move:
            engine = white_engine
            color = "White"
        else:
            engine = black_engine
            color = "Black"

        # Get move
        move = engine.select_move(board, depth=depth)

        if move is None:
            break

        # Make move
        move_str = board.move_to_algebraic(move)
        captured = board.make_move(move)
        move_list.append(move_str)

        if verbose:
            capture_str = f" x {captured}" if captured != '.' else ""
            print(f"Move {board.move_count}: {color} plays {move_str}{capture_str}")

            # Show board every 10 moves
            if board.move_count % 10 == 0:
                print(f"\nPosition after move {board.move_count}:")
                print(board.display())
                print(f"Evaluation: {board.evaluate():+.2f}")
                print()

    # Game over
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
        print(f"\nMove list: {' '.join(move_list[:20])}{'...' if len(move_list) > 20 else ''}")

    return {
        'winner': winner,
        'moves': board.move_count,
        'move_list': move_list,
        'final_eval': board.evaluate(),
        'white_evals': white_engine.evaluations,
        'black_evals': black_engine.evaluations
    }


def tournament(num_games: int = 5, depth: int = 2):
    """Run a tournament between two Genesis engines."""

    print("=" * 60)
    print("GENESIS ENGINE TOURNAMENT")
    print("=" * 60)
    print(f"\nGames: {num_games}")
    print(f"Search depth: {depth}")
    print(f"\nGAMMA constraint: {GAMMA:.6f}")
    print()

    # Create two engines with different random seeds
    engine_alpha = GenesisBrain("ALPHA (Seed: PHI)", seed=int(PHI * 1000))
    engine_beta = GenesisBrain("BETA (Seed: GAMMA)", seed=int(GAMMA * 10000))

    results = {'ALPHA': 0, 'BETA': 0, 'Draw': 0}

    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"GAME {game_num + 1} of {num_games}")
        print(f"{'='*60}")

        # Alternate colors
        if game_num % 2 == 0:
            white = engine_alpha
            black = engine_beta
            white_name = 'ALPHA'
            black_name = 'BETA'
        else:
            white = engine_beta
            black = engine_alpha
            white_name = 'BETA'
            black_name = 'ALPHA'

        # Reset engines
        engine_alpha.evaluations = 0
        engine_beta.evaluations = 0
        engine_alpha.d_phase = torch.zeros(64)
        engine_beta.d_phase = torch.zeros(64)

        print(f"White: {white_name} | Black: {black_name}")

        result = play_game(white, black, verbose=False, depth=depth)

        if result['winner'] == 'White':
            results[white_name] += 1
            print(f"Result: {white_name} wins in {result['moves']} moves")
        elif result['winner'] == 'Black':
            results[black_name] += 1
            print(f"Result: {black_name} wins in {result['moves']} moves")
        else:
            results['Draw'] += 1
            print(f"Result: Draw after {result['moves']} moves")

    print(f"\n{'='*60}")
    print("TOURNAMENT RESULTS")
    print(f"{'='*60}")
    print(f"\nALPHA: {results['ALPHA']} wins")
    print(f"BETA:  {results['BETA']} wins")
    print(f"Draws: {results['Draw']}")

    if results['ALPHA'] > results['BETA']:
        print(f"\nChampion: ALPHA (seeded with PHI)")
    elif results['BETA'] > results['ALPHA']:
        print(f"\nChampion: BETA (seeded with GAMMA)")
    else:
        print(f"\nResult: TIE!")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "   GENESIS ENGINE CHESS BATTLE   ".center(58) + "*")
    print("*" + "   Gamma-Constrained Neural Networks   ".center(58) + "*")
    print("*" + f"   Gamma = 1/(6*phi) = {GAMMA:.6f}   ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()

    # Create two Genesis engines
    engine_white = GenesisBrain("GENESIS-WHITE", seed=42)
    engine_black = GenesisBrain("GENESIS-BLACK", seed=74)  # N_EVO seed!

    # Play a single game (depth=1 for speed)
    result = play_game(engine_white, engine_black, verbose=True, depth=1)

    print("\n" + "=" * 60)
    print("BATTLE STATISTICS")
    print("=" * 60)
    print(f"Winner: {result['winner']}")
    print(f"Total moves: {result['moves']}")
    print(f"White evaluations: {result['white_evals']}")
    print(f"Black evaluations: {result['black_evals']}")
    print(f"Final material balance: {result['final_eval']:+.2f}")
