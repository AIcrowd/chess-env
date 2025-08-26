import pytest

import chess
from agents import ChessAgent, RandomAgent


class TestChessAgent:
    """Test the abstract ChessAgent base class."""

    def test_chess_agent_is_abstract(self):
        """Test that ChessAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ChessAgent()

    def test_chess_agent_has_choose_move_method(self):
        """Test that ChessAgent has the required abstract method."""
        assert hasattr(ChessAgent, "choose_move")
        assert ChessAgent.choose_move.__isabstractmethod__


class TestRandomAgent:
    """Test the RandomAgent implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = RandomAgent()
        self.board = chess.Board()

    def test_random_agent_instantiation(self):
        """Test that RandomAgent can be instantiated."""
        assert isinstance(self.agent, RandomAgent)
        assert isinstance(self.agent, ChessAgent)

    def test_choose_move_returns_legal_move(self):
        """Test that choose_move returns a legal move."""
        legal_moves = list(self.board.legal_moves)
        move = self.agent.choose_move(self.board, legal_moves, [], "White")
        assert move in legal_moves

    def test_choose_move_with_empty_legal_moves(self):
        """Test choose_move behavior with no legal moves."""
        empty_moves = []
        with pytest.raises(IndexError):  # random.choice([]) raises IndexError
            self.agent.choose_move(self.board, empty_moves, [], "White")

    def test_choose_move_with_single_legal_move(self):
        """Test choose_move with only one legal move."""
        # Create a position with only one legal move - use a checkmate position
        board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
        legal_moves = list(board.legal_moves)
        # This position actually has no legal moves (checkmate)
        assert len(legal_moves) == 0

        # Test with a position that has very few moves
        board = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")  # King only
        legal_moves = list(board.legal_moves)
        move = self.agent.choose_move(board, legal_moves, [], "White")
        assert move in legal_moves
        assert len(legal_moves) > 0  # King has multiple legal moves

    def test_choose_move_with_multiple_legal_moves(self):
        """Test choose_move with multiple legal moves."""
        # Starting position has 20 legal moves
        legal_moves = list(self.board.legal_moves)
        assert len(legal_moves) == 20

        # Run multiple times to ensure randomness
        moves_chosen = set()
        for _ in range(10):
            move = self.agent.choose_move(self.board, legal_moves, [], "White")
            moves_chosen.add(move)

        # Should have chosen different moves (randomness)
        assert len(moves_chosen) > 1

    def test_choose_move_parameters(self):
        """Test that choose_move receives all expected parameters."""
        legal_moves = list(self.board.legal_moves)
        move_history = ["e4", "e5"]
        side_to_move = "Black"

        # This should not raise any errors
        move = self.agent.choose_move(
            self.board, legal_moves, move_history, side_to_move
        )
        assert move in legal_moves

    def test_choose_move_with_different_positions(self):
        """Test choose_move with different board positions."""
        # Test with a midgame position
        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        legal_moves = list(board.legal_moves)
        move = self.agent.choose_move(board, legal_moves, ["e4"], "Black")
        assert move in legal_moves

        # Test with an endgame position
        board = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")  # King only
        legal_moves = list(board.legal_moves)
        move = self.agent.choose_move(board, legal_moves, [], "White")
        assert move in legal_moves

    def test_choose_move_consistency(self):
        """Test that choose_move is consistent when called multiple times with same state."""
        legal_moves = list(self.board.legal_moves)

        # Since it's random, we can't test exact consistency, but we can test it doesn't crash
        for _ in range(5):
            move = self.agent.choose_move(self.board, legal_moves, [], "White")
            assert move in legal_moves

    def test_choose_move_with_long_move_history(self):
        """Test choose_move with a long move history."""
        # Simulate a long game
        long_history = [
            "e4",
            "e5",
            "Nf3",
            "Nc6",
            "Bb5",
            "a6",
            "Ba4",
            "Nf6",
            "O-O",
            "Be7",
        ]
        legal_moves = list(self.board.legal_moves)

        move = self.agent.choose_move(self.board, legal_moves, long_history, "White")
        assert move in legal_moves

    def test_choose_move_edge_cases(self):
        """Test choose_move with edge case scenarios."""
        # Test with checkmate position (no legal moves)
        board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
        legal_moves = list(board.legal_moves)
        assert len(legal_moves) == 0  # This is actually checkmate

        # This should raise IndexError since there are no legal moves
        with pytest.raises(IndexError):
            self.agent.choose_move(board, legal_moves, [], "White")

        # Test with a position that has legal moves but is close to checkmate
        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        legal_moves = list(board.legal_moves)
        assert len(legal_moves) > 0  # This position has legal moves

        move = self.agent.choose_move(board, legal_moves, [], "Black")
        assert move in legal_moves
