"""
Tests for the new agent implementations.
"""

import pytest

import chess
from agents import FirstMoveAgent, LastMoveAgent


class TestFirstMoveAgent:
    """Test the FirstMoveAgent implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = FirstMoveAgent()
        self.board = chess.Board()
    
    def test_first_move_agent_instantiation(self):
        """Test that FirstMoveAgent can be instantiated."""
        assert isinstance(self.agent, FirstMoveAgent)
    
    def test_choose_move_returns_first_legal_move(self):
        """Test that choose_move returns the first legal move."""
        legal_moves = list(self.board.legal_moves)
        move = self.agent.choose_move(self.board, legal_moves, [], "White")
        assert move == legal_moves[0]
    
    def test_choose_move_consistency(self):
        """Test that choose_move always returns the same move for the same position."""
        legal_moves = list(self.board.legal_moves)
        
        # Should always return the first move
        for _ in range(5):
            move = self.agent.choose_move(self.board, legal_moves, [], "White")
            assert move == legal_moves[0]


class TestLastMoveAgent:
    """Test the LastMoveAgent implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = LastMoveAgent()
        self.board = chess.Board()
    
    def test_last_move_agent_instantiation(self):
        """Test that LastMoveAgent can be instantiated."""
        assert isinstance(self.agent, LastMoveAgent)
    
    def test_choose_move_returns_last_legal_move(self):
        """Test that choose_move returns the last legal move."""
        legal_moves = list(self.board.legal_moves)
        move = self.agent.choose_move(self.board, legal_moves, [], "White")
        assert move == legal_moves[-1]
    
    def test_choose_move_consistency(self):
        """Test that choose_move always returns the same move for the same position."""
        legal_moves = list(self.board.legal_moves)
        
        # Should always return the last move
        for _ in range(5):
            move = self.agent.choose_move(self.board, legal_moves, [], "White")
            assert move == legal_moves[-1]
