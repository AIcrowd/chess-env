import unittest
from unittest.mock import Mock, patch

import chess
from agents import RandomAgent
from env import ChessEnvironment


class TestChessEnvironment(unittest.TestCase):
    """Test cases for the ChessEnvironment class."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent1 = RandomAgent()
        self.agent2 = RandomAgent()
        self.env = ChessEnvironment(self.agent1, self.agent2)

    def test_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env.agent1)
        self.assertIsNotNone(self.env.agent2)
        self.assertEqual(self.env.max_moves, 200)
        self.assertEqual(self.env.time_limit, 10.0)
        self.assertIsInstance(self.env.board, chess.Board)
        self.assertEqual(len(self.env.move_history), 0)
        self.assertIsNone(self.env.game_result)

    def test_initialization_with_custom_fen(self):
        """Test environment initialization with custom FEN."""
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        env = ChessEnvironment(self.agent1, self.agent2, initial_fen=custom_fen)
        
        # FEN might be normalized by python-chess, so compare board positions
        expected_board = chess.Board(custom_fen)
        self.assertEqual(env.board.fen(), expected_board.fen())
        self.assertEqual(env.get_side_to_move(), "Black")

    def test_reset(self):
        """Test board reset functionality."""
        # Make some moves first
        self.env.board.push(chess.Move.from_uci("e2e4"))
        self.env.move_history = ["e2e4"]
        
        # Reset to starting position
        self.env.reset()
        self.assertEqual(self.env.get_fen(), chess.STARTING_FEN)
        self.assertEqual(len(self.env.move_history), 0)
        self.assertIsNone(self.env.game_result)

    def test_reset_with_custom_fen(self):
        """Test reset with custom FEN position."""
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        self.env.reset(custom_fen)
        
        # FEN might be normalized by python-chess, so compare board positions
        expected_board = chess.Board(custom_fen)
        self.assertEqual(self.env.board.fen(), expected_board.fen())
        self.assertEqual(self.env.get_side_to_move(), "Black")

    def test_get_legal_moves(self):
        """Test getting legal moves."""
        legal_moves = self.env.get_legal_moves()
        assert len(legal_moves) == 20  # Starting position has 20 legal moves
        assert all(isinstance(move, chess.Move) for move in legal_moves)

    def test_get_legal_moves_uci(self):
        """Test getting legal moves in UCI notation."""
        legal_moves_uci = self.env.get_legal_moves_uci()
        assert len(legal_moves_uci) == 20
        assert all(isinstance(move, str) for move in legal_moves_uci)
        assert "e2e4" in legal_moves_uci
        assert "g1f3" in legal_moves_uci

    def test_get_fen(self):
        """Test getting FEN notation."""
        fen = self.env.get_fen()
        assert fen == chess.STARTING_FEN
        assert isinstance(fen, str)

    def test_get_side_to_move(self):
        """Test getting whose turn it is."""
        # Starting position: White's turn
        assert self.env.get_side_to_move() == "White"

        # After a move, should be Black's turn
        self.env.board.push(chess.Move.from_uci("e2e4"))
        assert self.env.get_side_to_move() == "Black"

    def test_get_last_move(self):
        """Test getting the last move played."""
        # No moves yet
        assert self.env.get_last_move() is None

        # After a move
        self.env.board.push(chess.Move.from_uci("e2e4"))
        self.env.move_history = ["e2e4"]
        assert self.env.get_last_move() == "e2e4"

    def test_is_game_over(self):
        """Test game over detection."""
        # Starting position: game not over
        assert not self.env.is_game_over()

        # Checkmate position
        checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        self.env.reset(checkmate_fen)
        assert self.env.is_game_over()

    def test_get_game_result(self):
        """Test getting game result."""
        # Game not over
        assert self.env.get_game_result() is None

        # Checkmate position
        checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        self.env.reset(checkmate_fen)
        assert self.env.get_game_result() == "Black wins"

        # Stalemate position
        stalemate_fen = "k7/8/1K6/8/8/8/8/8 w - - 0 1"
        self.env.reset(stalemate_fen)
        assert self.env.get_game_result() == "Draw"

    def test_play_move_valid(self):
        """Test playing a valid move."""
        legal_moves = self.env.get_legal_moves()
        move = legal_moves[0]  # First legal move

        success = self.env.play_move(move)
        assert success
        assert len(self.env.move_history) == 1
        assert self.env.board.turn == chess.BLACK  # Should be Black's turn now

    def test_play_move_invalid(self):
        """Test playing an invalid move."""
        # Create an invalid move
        invalid_move = chess.Move.from_uci("e2e5")  # e2e5 is not legal from start

        success = self.env.play_move(invalid_move)
        assert not success
        assert len(self.env.move_history) == 0

    def test_play_agent_move_success(self):
        """Test successful agent move."""
        move = self.env.play_agent_move(self.agent1, "White")
        assert move is not None
        assert len(self.env.move_history) == 1
        assert self.env.board.turn == chess.BLACK

    def test_play_agent_move_failure(self):
        """Test agent move failure."""

        # Create a mock agent that returns invalid moves
        class MockAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                return chess.Move.from_uci("e2e5")  # Invalid move

        mock_agent = MockAgent()
        move = self.env.play_agent_move(mock_agent, "White")
        assert move is None
        assert len(self.env.move_history) == 0

    def test_play_agent_move_timeout(self):
        """Test agent move timeout."""
        # Create a mock agent that takes too long
        class SlowAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                import time
                time.sleep(0.1)  # Simulate slow response
                return super().choose_move(board, legal_moves, move_history, side_to_move)

        slow_agent = SlowAgent()
        # Set very short time limit
        self.env.time_limit = 0.05
        move = self.env.play_agent_move(slow_agent, "White")
        # Should still work but with warning
        assert move is not None

    def test_play_game_basic(self):
        """Test basic game play."""
        # Create a simple game with few moves
        env = ChessEnvironment(self.agent1, self.agent2, max_moves=5)
        result = env.play_game(verbose=False)
        
        assert "result" in result
        assert "moves_played" in result
        assert "move_history" in result
        assert "final_fen" in result
        assert "white_agent" in result
        assert "black_agent" in result
        assert "game_over_reason" in result

    def test_play_game_max_moves(self):
        """Test game ending due to max moves."""
        env = ChessEnvironment(self.agent1, self.agent2, max_moves=2)
        result = env.play_game(verbose=False)
        
        assert result["game_over_reason"] == "max_moves"
        assert result["moves_played"] == 2

    def test_play_game_checkmate(self):
        """Test game ending due to checkmate."""
        # Start from a position close to checkmate
        checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        env = ChessEnvironment(self.agent1, self.agent2, initial_fen=checkmate_fen)
        
        # Set a reasonable max_moves to allow checkmate to happen
        env.max_moves = 10
        result = env.play_game(verbose=False)
        
        # The game should complete, either with checkmate or max moves
        self.assertIsNotNone(result["result"])
        self.assertGreater(result["moves_played"], 0)
        
        # Check if it's a checkmate or max moves reached
        if "checkmate" in result["result"].lower() or "wins" in result["result"].lower():
            # Checkmate occurred
            pass
        elif "max moves" in result["result"].lower():
            # Max moves reached before checkmate
            pass
        else:
            # Unexpected result
            self.fail(f"Unexpected game result: {result['result']}")

    def test_get_pgn(self):
        """Test PGN generation."""
        # Play a few moves
        self.env.board.push(chess.Move.from_uci("e2e4"))
        self.env.board.push(chess.Move.from_uci("e7e5"))
        self.env.move_history = ["e2e4", "e7e5"]
        
        pgn = self.env.get_pgn()
        assert "[Event" in pgn
        assert "[White" in pgn
        assert "[Black" in pgn
        assert "e4" in pgn  # Should be converted to SAN for PGN
        assert "e5" in pgn  # Should be converted to SAN for PGN

    def test_export_pgn_file(self):
        """Test PGN file export."""
        # Play a few moves
        self.env.board.push(chess.Move.from_uci("e2e4"))
        self.env.board.push(chess.Move.from_uci("e7e5"))
        self.env.move_history = ["e2e4", "e7e5"]
        
        # Test export
        success = self.env.export_pgn_file("test_game")
        assert success
        
        # Check file was created
        import os
        assert os.path.exists("test_game.pgn")
        
        # Clean up
        os.remove("test_game.pgn")

    def test_export_pgn_file_with_metadata(self):
        """Test PGN file export with metadata."""
        # Play a few moves
        self.env.board.push(chess.Move.from_uci("e2e4"))
        self.env.board.push(chess.Move.from_uci("e7e5"))
        self.env.move_history = ["e2e4", "e7e5"]
        
        # Test export with metadata
        success = self.env.export_pgn_file("test_game_meta", include_metadata=True)
        assert success
        
        # Check file was created
        import os
        assert os.path.exists("test_game_meta.pgn")
        
        # Check metadata content
        with open("test_game_meta.pgn", 'r') as f:
            content = f.read()
            assert '[InitialFEN' in content
            assert '[FinalFEN' in content
        
        # Clean up
        os.remove("test_game_meta.pgn")

    def test_display_board(self):
        """Test board display functionality."""
        display = self.env.display_board(clean=False)  # Use non-clean mode for testing
        assert isinstance(display, str)
        assert len(display) > 0
    
    def test_display_board_highlight_last_move(self):
        """Test board display with last move highlighting."""
        # Play a move
        self.env.board.push(chess.Move.from_uci("e2e4"))
        self.env.move_history = ["e2e4"]
    
        display = self.env.display_board(highlight_last_move=True, clean=False)  # Use non-clean mode for testing
        assert isinstance(display, str)
        assert len(display) > 0

    def test_display_game_state(self):
        """Test game state display functionality."""
        display = self.env.display_game_state()
        assert isinstance(display, str)
        assert len(display) > 0
        assert "Side to move: White" in display

    def test_display_position_analysis(self):
        """Test position analysis display functionality."""
        display = self.env.display_position_analysis()
        assert isinstance(display, str)
        assert len(display) > 0
        assert "Position Analysis:" in display
        assert "White material:" in display
        assert "Black material:" in display

    def test_display_move_sequence(self):
        """Test move sequence display functionality."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3")
        ]
        
        display = self.env.display_move_sequence(moves)
        assert isinstance(display, str)
        assert len(display) > 0
        assert "Move 1: e2e4" in display  # UCI notation
        assert "Move 2: e7e5" in display  # UCI notation
        assert "Move 3: g1f3" in display  # UCI notation

    def test_display_move_sequence_with_start_fen(self):
        """Test move sequence display with custom starting position."""
        moves = [
            chess.Move.from_uci("g8f6"),
            chess.Move.from_uci("d2d4")
        ]
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
        display = self.env.display_move_sequence(moves, start_fen)
        assert isinstance(display, str)
        assert len(display) > 0
        # Check for professional style output (default)
        assert "🎯 MOVE SEQUENCE DISPLAY 🎯" in display
        assert "📍 Initial Position:" in display

    def test_set_renderer_options(self):
        """Test renderer options configuration."""
        # Test setting various options
        self.env.set_renderer_options(show_coordinates=False)
        self.env.set_renderer_options(show_move_numbers=True)
        self.env.set_renderer_options(empty_square_char=".")
        self.env.set_renderer_options(use_rich=False)
        
        # Verify options were set (we can't easily test the internal state,
        # but we can verify the methods don't raise errors)
        display = self.env.display_board()
        assert isinstance(display, str)

    def test_custom_fen_initialization(self):
        """Test environment initialization with various custom FEN positions."""
        test_positions = [
            ("Starting position", chess.STARTING_FEN),
            ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            ("King and pawn endgame", "8/8/8/8/8/8/4P3/4K3 w - - 0 1"),
            ("Fool's mate position", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
        ]
        
        for name, fen in test_positions:
            with self.subTest(name=name):
                env = ChessEnvironment(self.agent1, self.agent2, initial_fen=fen)
                
                # FEN might be normalized by python-chess, so compare board positions
                expected_board = chess.Board(fen)
                self.assertEqual(env.board.fen(), expected_board.fen())
                self.assertEqual(env._initial_fen, fen)

    def test_move_history_consistency(self):
        """Test that move history is consistent throughout the game."""
        # Play a few moves
        moves = ["e2e4", "e7e5", "g1f3"]
        
        for move_str in moves:
            move = chess.Move.from_uci(move_str)
            success = self.env.play_move(move)
            assert success
        
        # Check move history
        assert len(self.env.move_history) == 3
        assert self.env.move_history == moves
        
        # Check that the board state matches the move history
        temp_board = chess.Board()
        for move_str in moves:
            move = chess.Move.from_uci(move_str)
            temp_board.push(move)
        
        assert self.env.get_fen() == temp_board.fen()

    def test_legal_moves_consistency(self):
        """Test that legal moves are consistent with board state."""
        # Get legal moves from environment
        env_legal_moves = set(move.uci() for move in self.env.get_legal_moves())
        
        # Get legal moves directly from board
        board_legal_moves = set(move.uci() for move in self.env.board.legal_moves)
        
        # They should be identical
        assert env_legal_moves == board_legal_moves

    def test_agent_interface_consistency(self):
        """Test that agents receive consistent information."""
        # Mock agent to capture what it receives
        received_info = {}
        
        class TestAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                received_info['board'] = board
                received_info['legal_moves'] = legal_moves
                received_info['move_history'] = move_history
                received_info['side_to_move'] = side_to_move
                return super().choose_move(board, legal_moves, move_history, side_to_move)
        
        test_agent = TestAgent()
        
        # Play a move
        self.env.play_agent_move(test_agent, "White")
        
        # Check that agent received correct information
        assert received_info['board'] == self.env.board
        assert received_info['side_to_move'] == "White"
        assert isinstance(received_info['legal_moves'], list)
        assert isinstance(received_info['move_history'], list)


if __name__ == "__main__":
    unittest.main()
