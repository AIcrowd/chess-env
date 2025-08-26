"""
First move chess agent implementation.

This agent always chooses the first available legal move.
"""

from typing import List

import chess

from .base import ChessAgent


class FirstMoveAgent(ChessAgent):
    """Agent that always chooses the first available legal move."""
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move | None, str | None]:
        """
        Choose the first available legal move.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: The first legal move from the list, or None to resign
            - optional_comment: None (first move strategy is self-explanatory)
            
        Raises:
            IndexError: If no legal moves are available
        """
        move = legal_moves[0]
        return move, None
