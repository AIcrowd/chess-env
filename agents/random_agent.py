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
    ) -> chess.Move:
        """
        Choose a random move from the legal moves.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in SAN notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            A randomly chosen legal move
            
        Raises:
            IndexError: If no legal moves are available
        """
        return random.choice(legal_moves)
