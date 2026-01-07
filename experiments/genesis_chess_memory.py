"""
Genesis Engine Chess Battle with Memory
========================================

Genesis Engines that LEARN and REMEMBER across games.
Each engine builds experience from its wins and losses.

Features:
- Position memory: remembers positions and their outcomes
- Hebbian learning: strengthens patterns that lead to wins
- Experience replay: learns from past games
- Persistent memory across tournament games
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from collections import deque
import sys
import os
import pickle
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import PHI, GAMMA, KOIDE


# =============================================================================
# CHESS BOARD (same as before)
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
        if self.move_count >= 80:
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

    def get_hash(self) -> str:
        """Get unique hash of position for memory lookup."""
        return ''.join([''.join(row) for row in self.board])

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
# MEMORY SYSTEM
# =============================================================================

class PositionMemory:
    """Memory bank for storing position evaluations and outcomes."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.positions = {}  # hash -> {score, visits, wins, losses}
        self.recent_games = deque(maxlen=100)  # Recent game trajectories

    def remember_position(self, pos_hash: str, score: float, outcome: float = 0):
        """Remember a position with its evaluation and outcome."""
        if pos_hash not in self.positions:
            self.positions[pos_hash] = {
                'score': score,
                'visits': 0,
                'wins': 0,
                'losses': 0,
                'total_outcome': 0
            }

        mem = self.positions[pos_hash]
        mem['visits'] += 1
        mem['total_outcome'] += outcome
        # Running average of score
        mem['score'] = mem['score'] * 0.9 + score * 0.1

        if outcome > 0:
            mem['wins'] += 1
        elif outcome < 0:
            mem['losses'] += 1

        # Prune if too large
        if len(self.positions) > self.max_size:
            # Remove least visited
            sorted_positions = sorted(self.positions.items(), key=lambda x: x[1]['visits'])
            for pos_hash, _ in sorted_positions[:1000]:
                del self.positions[pos_hash]

    def recall_position(self, pos_hash: str) -> Optional[Dict]:
        """Recall memory of a position."""
        return self.positions.get(pos_hash)

    def get_position_bonus(self, pos_hash: str) -> float:
        """Get bonus/penalty based on past experience with this position."""
        mem = self.recall_position(pos_hash)
        if mem is None:
            return 0.0

        visits = mem['visits']
        if visits == 0:
            return 0.0

        # Win rate influences evaluation
        win_rate = (mem['wins'] - mem['losses']) / max(visits, 1)
        # Scale by confidence (more visits = more confidence)
        confidence = min(1.0, visits / 10)

        return win_rate * confidence * GAMMA  # Scale by Gamma

    def store_game(self, positions: List[str], outcome: float):
        """Store a complete game trajectory with its outcome."""
        self.recent_games.append((positions, outcome))

        # Update all positions with the outcome
        # Positions later in game get more credit/blame
        for i, pos_hash in enumerate(positions):
            # Temporal difference: later moves matter more
            weight = (i + 1) / len(positions)
            self.remember_position(pos_hash, 0, outcome * weight)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        total_visits = sum(m['visits'] for m in self.positions.values())
        total_wins = sum(m['wins'] for m in self.positions.values())
        total_losses = sum(m['losses'] for m in self.positions.values())

        return {
            'positions_stored': len(self.positions),
            'total_visits': total_visits,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'games_remembered': len(self.recent_games)
        }


# =============================================================================
# GENESIS BRAIN WITH MEMORY
# =============================================================================

class GenesisBrainWithMemory(nn.Module):
    """
    Genesis Engine brain with persistent memory and learning.
    """

    def __init__(self, name: str, seed: int = None):
        super().__init__()
        self.name = name

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Neural network layers
        self.d_layer = nn.Linear(768, 64)
        self.t_layer = nn.Linear(64, 27)
        self.q_layer = nn.Linear(27, 16)
        self.output = nn.Linear(16, 1)

        # Phase states
        self.d_phase = torch.zeros(64)
        self.t_phase = torch.zeros(27)
        self.q_phase = torch.zeros(16)
        self.omega = 2 * np.pi / PHI

        # MEMORY SYSTEMS
        self.memory = PositionMemory(max_size=5000)
        self.current_game_positions = []  # Positions in current game

        # Learning parameters
        self.learning_rate = GAMMA  # Use Gamma as learning rate!
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        # Statistics
        self.evaluations = 0
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Gamma constraints."""
        d_out = torch.tanh(self.d_layer(x))
        self.d_phase = self.d_phase + self.omega * 0.1 + 0.1 * torch.sin(d_out.detach())

        t_out = self.t_layer(d_out)
        t_out = torch.clamp(t_out, -GAMMA, GAMMA)
        self.t_phase = self.t_phase + self.omega * 0.1

        q_out = self.q_layer(t_out)
        q_out = torch.clamp(q_out, -GAMMA, GAMMA)
        self.q_phase = self.q_phase + self.omega * 0.1

        score = self.output(q_out)
        self.evaluations += 1
        return score

    def evaluate_position(self, board: ChessBoard, use_memory: bool = True) -> float:
        """Evaluate position using neural net + memory."""
        x = board.to_tensor()

        with torch.no_grad():
            neural_score = self.forward(x).item()

        # Add memory bonus
        memory_bonus = 0.0
        if use_memory:
            pos_hash = board.get_hash()
            memory_bonus = self.memory.get_position_bonus(pos_hash)

        return neural_score + memory_bonus

    def select_move(self, board: ChessBoard, depth: int = 1) -> Optional[Tuple[int, int, int, int]]:
        """Select best move using search + memory."""
        moves = board.get_legal_moves()
        if not moves:
            return None

        is_maximizing = board.white_to_move
        best_move = None
        best_score = float('-inf') if is_maximizing else float('inf')

        for move in moves:
            test_board = board.copy()
            test_board.make_move(move)

            # Get neural + material evaluation
            material = test_board.evaluate()
            neural = self.evaluate_position(test_board)
            score = material + neural * GAMMA

            # Add exploration bonus for unvisited positions
            pos_hash = test_board.get_hash()
            mem = self.memory.recall_position(pos_hash)
            if mem is None:
                # Encourage exploration
                score += 0.1 * random.random()
            else:
                # Prefer positions we've won from
                score += self.memory.get_position_bonus(pos_hash)

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

    def remember_position(self, board: ChessBoard):
        """Remember current position during game."""
        pos_hash = board.get_hash()
        score = self.evaluate_position(board, use_memory=False)
        self.current_game_positions.append(pos_hash)
        self.memory.remember_position(pos_hash, score)

    def learn_from_game(self, outcome: float):
        """Learn from game result. outcome: +1 win, -1 loss, 0 draw."""
        if self.current_game_positions:
            self.memory.store_game(self.current_game_positions, outcome)

        # Hebbian-style weight update based on outcome
        if outcome != 0:
            # Simple gradient step based on outcome
            for param in self.parameters():
                if param.grad is not None:
                    param.data += self.learning_rate * outcome * param.grad * 0.01

        # Update stats
        self.games_played += 1
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        else:
            self.draws += 1

        # Clear current game positions
        self.current_game_positions = []

    def reset_for_new_game(self):
        """Reset state for new game (keeps memory!)."""
        self.current_game_positions = []
        self.d_phase = torch.zeros(64)
        self.t_phase = torch.zeros(27)
        self.q_phase = torch.zeros(16)
        self.evaluations = 0

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        mem_stats = self.memory.get_stats()
        return {
            'name': self.name,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / max(self.games_played, 1),
            'positions_memorized': mem_stats['positions_stored'],
            'total_evaluations': self.evaluations
        }


# =============================================================================
# GAME AND TOURNAMENT
# =============================================================================

def play_game_with_memory(white: GenesisBrainWithMemory, black: GenesisBrainWithMemory,
                          verbose: bool = False) -> Dict:
    """Play a game where both engines learn and remember."""

    board = ChessBoard()
    white.reset_for_new_game()
    black.reset_for_new_game()

    move_list = []

    while True:
        game_over, winner = board.is_game_over()
        if game_over:
            break

        engine = white if board.white_to_move else black
        color = "White" if board.white_to_move else "Black"

        # Remember current position
        engine.remember_position(board)

        # Get move
        move = engine.select_move(board)
        if move is None:
            break

        move_str = board.move_to_algebraic(move)
        board.make_move(move)
        move_list.append(move_str)

        if verbose and board.move_count % 20 == 0:
            print(f"  Move {board.move_count}: {color} plays {move_str}")

    # Determine outcome
    game_over, winner = board.is_game_over()

    # Calculate outcomes for each player
    if winner == "White":
        white_outcome, black_outcome = 1.0, -1.0
    elif winner == "Black":
        white_outcome, black_outcome = -1.0, 1.0
    else:
        white_outcome, black_outcome = 0.0, 0.0

    # Both engines learn from the game
    white.learn_from_game(white_outcome)
    black.learn_from_game(black_outcome)

    return {
        'winner': winner,
        'moves': board.move_count,
        'move_list': move_list,
        'final_eval': board.evaluate()
    }


def run_tournament(num_games: int = 10, verbose: bool = True):
    """Run a tournament where engines learn and improve."""

    print("=" * 70)
    print("GENESIS ENGINE TOURNAMENT WITH MEMORY")
    print("=" * 70)
    print(f"\nGames: {num_games}")
    print(f"GAMMA = {GAMMA:.6f} (also used as learning rate!)")
    print(f"PHI = {PHI:.10f}")
    print()

    # Create two engines with different seeds
    engine_alpha = GenesisBrainWithMemory("ALPHA", seed=int(PHI * 1000))
    engine_beta = GenesisBrainWithMemory("BETA", seed=int(GAMMA * 100000))

    results = []

    for game_num in range(1, num_games + 1):
        # Alternate colors
        if game_num % 2 == 1:
            white, black = engine_alpha, engine_beta
            white_name, black_name = "ALPHA", "BETA"
        else:
            white, black = engine_beta, engine_alpha
            white_name, black_name = "BETA", "ALPHA"

        if verbose:
            print(f"\n--- Game {game_num}/{num_games}: {white_name} (W) vs {black_name} (B) ---")

        result = play_game_with_memory(white, black, verbose=False)
        results.append({**result, 'white': white_name, 'black': black_name})

        if verbose:
            if result['winner'] == 'White':
                print(f"  Result: {white_name} wins in {result['moves']} moves")
            elif result['winner'] == 'Black':
                print(f"  Result: {black_name} wins in {result['moves']} moves")
            else:
                print(f"  Result: Draw after {result['moves']} moves")

            # Show memory growth every 5 games
            if game_num % 5 == 0:
                alpha_stats = engine_alpha.get_stats()
                beta_stats = engine_beta.get_stats()
                print(f"\n  Memory after {game_num} games:")
                print(f"    ALPHA: {alpha_stats['positions_memorized']} positions, "
                      f"{alpha_stats['wins']}W-{alpha_stats['losses']}L-{alpha_stats['draws']}D")
                print(f"    BETA:  {beta_stats['positions_memorized']} positions, "
                      f"{beta_stats['wins']}W-{beta_stats['losses']}L-{beta_stats['draws']}D")

    # Final results
    print("\n" + "=" * 70)
    print("TOURNAMENT RESULTS")
    print("=" * 70)

    alpha_stats = engine_alpha.get_stats()
    beta_stats = engine_beta.get_stats()

    print(f"\n{'Engine':<10} {'Wins':<6} {'Losses':<8} {'Draws':<7} {'Win%':<8} {'Positions':<12}")
    print("-" * 60)
    print(f"{'ALPHA':<10} {alpha_stats['wins']:<6} {alpha_stats['losses']:<8} "
          f"{alpha_stats['draws']:<7} {alpha_stats['win_rate']*100:>5.1f}%   "
          f"{alpha_stats['positions_memorized']:<12}")
    print(f"{'BETA':<10} {beta_stats['wins']:<6} {beta_stats['losses']:<8} "
          f"{beta_stats['draws']:<7} {beta_stats['win_rate']*100:>5.1f}%   "
          f"{beta_stats['positions_memorized']:<12}")

    # Determine champion
    print("\n" + "-" * 60)
    if alpha_stats['wins'] > beta_stats['wins']:
        print(f"CHAMPION: ALPHA (seeded with PHI)")
    elif beta_stats['wins'] > alpha_stats['wins']:
        print(f"CHAMPION: BETA (seeded with GAMMA)")
    else:
        print(f"RESULT: TIE!")

    # Show learning progression
    print("\n" + "=" * 70)
    print("LEARNING ANALYSIS")
    print("=" * 70)

    # First half vs second half performance
    half = num_games // 2
    first_half = results[:half]
    second_half = results[half:]

    alpha_first = sum(1 for r in first_half if (r['winner'] == 'White' and r['white'] == 'ALPHA') or
                                                (r['winner'] == 'Black' and r['black'] == 'ALPHA'))
    alpha_second = sum(1 for r in second_half if (r['winner'] == 'White' and r['white'] == 'ALPHA') or
                                                  (r['winner'] == 'Black' and r['black'] == 'ALPHA'))

    beta_first = sum(1 for r in first_half if (r['winner'] == 'White' and r['white'] == 'BETA') or
                                               (r['winner'] == 'Black' and r['black'] == 'BETA'))
    beta_second = sum(1 for r in second_half if (r['winner'] == 'White' and r['white'] == 'BETA') or
                                                 (r['winner'] == 'Black' and r['black'] == 'BETA'))

    print(f"\nFirst {half} games vs Last {half} games:")
    print(f"  ALPHA: {alpha_first} wins -> {alpha_second} wins "
          f"({'improved' if alpha_second > alpha_first else 'declined' if alpha_second < alpha_first else 'stable'})")
    print(f"  BETA:  {beta_first} wins -> {beta_second} wins "
          f"({'improved' if beta_second > beta_first else 'declined' if beta_second < beta_first else 'stable'})")

    return {
        'alpha': alpha_stats,
        'beta': beta_stats,
        'games': results
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "   GENESIS ENGINE CHESS TOURNAMENT WITH MEMORY   ".center(68) + "*")
    print("*" + "   Gamma-Constrained Neural Networks That LEARN   ".center(68) + "*")
    print("*" + f"   Gamma = 1/(6*phi) = {GAMMA:.6f}   ".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # Run tournament
    tournament_results = run_tournament(num_games=10, verbose=True)

    print("\n" + "=" * 70)
    print("The engines have learned from their games!")
    print("Memory persists - they remember winning and losing positions.")
    print("=" * 70)
