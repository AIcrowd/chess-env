"""
Base chess agent abstract class.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

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
    ) -> Tuple[Union[chess.Move, None], Union[str, None]]:
        """
        Choose a move from the given legal moves.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: The chosen chess move, or None to resign
            - optional_comment: Optional comment explaining the move or resignation reason
        """
        pass
