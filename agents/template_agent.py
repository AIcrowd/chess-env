"""
Template for implementing new chess agents.

Copy this file and modify it to create your own chess agent.
"""

from typing import List

import chess

from .base import ChessAgent


class TemplateAgent(ChessAgent):
    """
    Template agent - replace this with your agent implementation.
    
    This is a starting point for implementing new chess agents.
    You can implement various strategies like:
    - Minimax with alpha-beta pruning
    - Monte Carlo Tree Search (MCTS)
    - Neural network evaluation
    - Opening book moves
    - Endgame tablebase lookups
    """
    
    def __init__(self, **kwargs):
        """
        Initialize your agent with any parameters it needs.
        
        Args:
            **kwargs: Any configuration parameters for your agent
        """
        # Add your initialization code here
        pass
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> chess.Move:
        """
        Choose a move using your agent's strategy.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in SAN notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            The chosen chess move
            
        Raises:
            IndexError: If no legal moves are available
        """
        # TODO: Implement your move selection logic here
        
        # Example: Always choose the first legal move
        # (Replace this with your actual strategy)
        return legal_moves[0]
        
        # Example: Evaluate all moves and choose the best one
        # best_move = None
        # best_score = float('-inf')
        # for move in legal_moves:
        #     # Make the move temporarily
        #     board.push(move)
        #     score = self.evaluate_position(board)
        #     board.pop()
        #     
        #     if score > best_score:
        #         best_score = score
        #         best_move = move
        # 
        # return best_move
    
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
