"""
First Move Agent implementation.

This agent always chooses the first legal move available.
"""

from typing import List

import chess

from .base import ChessAgent


class FirstMoveAgent(ChessAgent):
    """Agent that always chooses the first legal move."""
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> chess.Move:
        """
        Choose the first legal move.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in SAN notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            The first legal move
            
        Raises:
            IndexError: If no legal moves are available
        """
        return legal_moves[0]
