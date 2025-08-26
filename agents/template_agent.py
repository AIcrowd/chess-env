"""
Template chess agent implementation.

This is a template for implementing custom chess agents.
Copy this file and modify the choose_move method to implement your strategy.
"""

from typing import List

import chess

from .base import ChessAgent


class TemplateAgent(ChessAgent):
    """
    Template agent that you can extend to implement your own chess strategy.
    
    To use this template:
    1. Copy this file to a new file (e.g., my_agent.py)
    2. Rename the class to something descriptive (e.g., MyAgent)
    3. Implement the choose_move method with your strategy
    4. Import and use your agent in your code
    """
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move | None, str | None]:
        """
        Choose a move based on your custom strategy.
        
        This is where you implement your chess logic. You have access to:
        - board: The current chess position
        - legal_moves: List of all legal moves in this position
        - move_history: List of moves played so far (in UCI notation)
        - side_to_move: Which side you're playing ('White' or 'Black')
        
        Args:
            board: Current chess board state
        legal_moves: List of legal moves available
        move_history: List of moves played so far (in UCI notation)
        side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: The chosen chess move, or None to resign
            - optional_comment: Optional comment explaining the move strategy or resignation
            
        Example strategies you could implement:
        - Material counting (evaluate piece values)
        - Position evaluation (control of center, pawn structure)
        - Opening book moves
        - Endgame tablebase lookups
        - Machine learning model predictions
        """
        # TODO: Implement your chess strategy here!
        # For now, just return the first legal move (like FirstMoveAgent)
        move = legal_moves[0]
        comment = "Template agent - using first available move"
        return move, comment
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Evaluation score (positive favors White, negative favors Black)
        """
        # TODO: Implement your position evaluation logic
        # This could be:
        # - Material counting
        # - Piece-square tables
        # - Neural network evaluation
        # - Engine evaluation
        
        return 0.0  # Replace with actual evaluation
