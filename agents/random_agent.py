"""
Random chess agent implementation.

This agent chooses moves randomly from the available legal moves.
"""

import random
from typing import List

import chess

from .base import ChessAgent


class RandomAgent(ChessAgent):
    """Simple agent that chooses random legal moves."""
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move, str | None]:
        """
        Choose a random move from the legal moves.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: A randomly chosen legal move
            - optional_comment: None (random moves don't need explanation)
            
        Raises:
            IndexError: If no legal moves are available
        """
        move = random.choice(legal_moves)
        return move, None
