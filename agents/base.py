"""
Base chess agent abstract class.
"""

from abc import ABC, abstractmethod
from typing import List

import chess


class ChessAgent(ABC):
    """Abstract base class for chess agents."""
    
    @abstractmethod
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> chess.Move:
        """
        Choose a move from the given legal moves.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in SAN notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            The chosen chess move
        """
        pass
