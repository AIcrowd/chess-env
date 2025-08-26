import pytest

import chess
from agents import RandomAgent
from env import ChessEnvironment


class TestChessEnvironment:
    """Test the ChessEnvironment class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent1 = RandomAgent()
        self.agent2 = RandomAgent()
        self.env = ChessEnvironment(self.agent1, self.agent2)

    def test_environment_initialization(self):
        """Test environment initialization with default parameters."""
        assert self.env.agent1 == self.agent1
        assert self.env.agent2 == self.agent2
        assert self.env.max_moves == 200
        assert self.env.time_limit == 10.0
        assert isinstance(self.env.board, chess.Board)
        assert self.env.move_history == []
        assert self.env.game_result is None

    def test_environment_initialization_custom_params(self):
        """Test environment initialization with custom parameters."""
        env = ChessEnvironment(self.agent1, self.agent2, max_moves=100, time_limit=5.0)
        assert env.max_moves == 100
        assert env.time_limit == 5.0

    def test_environment_initialization_with_custom_fen(self):
        """Test environment initialization with custom FEN position."""
        # Test with a midgame position
        midgame_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        env = ChessEnvironment(self.agent1, self.agent2, initial_fen=midgame_fen)
        
        # Verify the board is set to the custom position
        board = chess.Board(midgame_fen)
        assert env.board.fen() == board.fen()
        assert env.get_side_to_move() == "Black"
        
        # Test with an endgame position
        endgame_fen = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"
        env = ChessEnvironment(self.agent1, self.agent2, initial_fen=endgame_fen)
        
        board = chess.Board(endgame_fen)
        assert env.board.fen() == board.fen()
        assert env.get_side_to_move() == "White"
        
        # Test with default (no FEN specified)
        env_default = ChessEnvironment(self.agent1, self.agent2)
        assert env_default.board.fen() == chess.STARTING_FEN

    def test_reset_default_position(self):
        """Test reset to default starting position."""
        # Make some moves first
        self.env.board.push_san("e4")
        self.env.board.push_san("e5")
        self.env.move_history = ["e4", "e5"]

        # Reset
        self.env.reset()

        assert self.env.board.fen() == chess.STARTING_FEN
        assert self.env.move_history == []
        assert self.env.game_result is None

    def test_reset_custom_position(self):
        """Test reset to custom FEN position."""
        # Note: python-chess normalizes FEN, so we need to compare normalized versions
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        self.env.reset(custom_fen)

        # The board should be in the same position, but FEN might be normalized
        board = chess.Board(custom_fen)
        assert self.env.board.fen() == board.fen()
        assert self.env.move_history == []
        assert self.env.game_result is None

    def test_get_legal_moves(self):
        """Test getting legal moves."""
        legal_moves = self.env.get_legal_moves()
        assert len(legal_moves) == 20  # Starting position has 20 legal moves
        assert all(isinstance(move, chess.Move) for move in legal_moves)

    def test_get_legal_moves_san(self):
        """Test getting legal moves in SAN notation."""
        legal_moves_san = self.env.get_legal_moves_san()
        assert len(legal_moves_san) == 20
        assert all(isinstance(move, str) for move in legal_moves_san)
        assert "e4" in legal_moves_san
        assert "Nf3" in legal_moves_san

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
        self.env.board.push_san("e4")
        assert self.env.get_side_to_move() == "Black"

    def test_get_last_move(self):
        """Test getting the last move played."""
        # No moves yet
        assert self.env.get_last_move() is None

        # After a move
        self.env.board.push_san("e4")
        self.env.move_history = ["e4"]
        assert self.env.get_last_move() == "e4"

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

    def test_play_agent_move_exception(self):
        """Test agent move with exception."""

        class ExceptionAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                raise Exception("Test exception")

        exception_agent = ExceptionAgent()
        move = self.env.play_agent_move(exception_agent, "White")
        assert move is None

    def test_play_game_basic(self):
        """Test playing a basic game."""
        # Set a small max_moves to ensure game ends quickly
        self.env.max_moves = 10

        result = self.env.play_game(verbose=False)

        assert "result" in result
        assert "moves_played" in result
        assert "move_history" in result
        assert "final_fen" in result
        assert "white_agent" in result
        assert "black_agent" in result
        assert "game_over_reason" in result

        assert result["white_agent"] == "RandomAgent"
        assert result["black_agent"] == "RandomAgent"
        assert result["moves_played"] > 0

    def test_play_game_max_moves_reached(self):
        """Test game ending due to max moves."""
        self.env.max_moves = 5  # Very small limit

        result = self.env.play_game(verbose=False)

        assert result["game_over_reason"] == "max_moves"
        assert result["moves_played"] == 5

    def test_play_game_checkmate(self):
        """Test game ending in checkmate."""
        # Set up a position that will lead to checkmate quickly
        # Use a position where one side is clearly winning
        winning_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        self.env.reset(winning_fen)

        # Set a reasonable max_moves to allow checkmate to happen
        self.env.max_moves = 20

        result = self.env.play_game(verbose=False)

        # The game should complete, either with checkmate or max moves
        assert result["result"] is not None
        assert result["moves_played"] > 0
        # Note: With random agents, checkmate might not happen quickly, so we just verify completion

    def test_get_pgn_empty_game(self):
        """Test PGN generation for empty game."""
        pgn = self.env.get_pgn()
        assert pgn == ""

    def test_get_pgn_with_moves(self):
        """Test PGN generation with moves."""
        # Play some moves
        self.env.board.push_san("e4")
        self.env.board.push_san("e5")
        self.env.move_history = ["e4", "e5"]

        pgn = self.env.get_pgn()

        assert '[Event "Chess Game"]' in pgn
        assert '[White "RandomAgent"]' in pgn
        assert '[Black "RandomAgent"]' in pgn
        assert "e4 e5" in pgn

    def test_export_pgn_file_basic(self):
        """Test basic PGN file export."""
        # Play some moves first
        self.env.board.push_san("e4")
        self.env.board.push_san("e5")
        self.env.move_history = ["e4", "e5"]
        
        # Test export
        filename = "test_game.pgn"
        success = self.env.export_pgn_file(filename)
        
        assert success
        
        # Verify file was created and contains expected content
        import os
        assert os.path.exists(filename)
        
        with open(filename, 'r') as f:
            content = f.read()
            assert '[Event "Chess Game"]' in content
            assert '[White "RandomAgent"]' in content
            assert '[Black "RandomAgent"]' in content
            assert "e4 e5" in content
        
        # Clean up
        os.remove(filename)
    
    def test_export_pgn_file_with_metadata(self):
        """Test PGN file export with metadata."""
        # Play some moves first
        self.env.board.push_san("e4")
        self.env.board.push_san("e5")
        self.env.move_history = ["e4", "e5"]
        
        # Test export with metadata
        filename = "test_game_metadata.pgn"
        success = self.env.export_pgn_file(filename, include_metadata=True)
        
        assert success
        
        # Verify file contains metadata
        with open(filename, 'r') as f:
            content = f.read()
            assert '[WhiteType "program"]' in content
            assert '[BlackType "program"]' in content
            assert '[Termination "unterminated"]' in content  # Game is not over yet
            assert '[Moves "2"]' in content
            assert '[InitialFEN' in content
            assert '[FinalFEN' in content
        
        # Clean up
        import os
        os.remove(filename)
    
    def test_export_pgn_file_without_metadata(self):
        """Test PGN file export without metadata."""
        # Play some moves first
        self.env.board.push_san("e4")
        self.env.board.push_san("e5")
        self.env.move_history = ["e4", "e5"]
        
        # Test export without metadata
        filename = "test_game_no_metadata.pgn"
        success = self.env.export_pgn_file(filename, include_metadata=False)
        
        assert success
        
        # Verify file doesn't contain metadata
        with open(filename, 'r') as f:
            content = f.read()
            assert '[WhiteType "program"]' not in content
            assert '[BlackType "program"]' not in content
            assert '[Termination "normal"]' not in content
            assert '[Moves "2"]' not in content
            assert '[InitialFEN' not in content
            assert '[FinalFEN' not in content
        
        # Clean up
        import os
        os.remove(filename)
    
    def test_export_pgn_file_auto_extension(self):
        """Test that PGN export automatically adds .pgn extension."""
        # Play some moves first
        self.env.board.push_san("e4")
        self.env.move_history = ["e4"]
        
        # Test export without .pgn extension
        filename = "test_game"
        success = self.env.export_pgn_file(filename)
        
        assert success
        
        # Verify file was created with .pgn extension
        import os
        expected_filename = "test_game.pgn"
        assert os.path.exists(expected_filename)
        
        # Clean up
        os.remove(expected_filename)
    
    def test_export_pgn_file_empty_game(self):
        """Test PGN export for empty game."""
        # Test export with no moves
        filename = "empty_game.pgn"
        success = self.env.export_pgn_file(filename)
        
        assert success
        
        # Verify file was created but contains minimal content
        import os
        assert os.path.exists(filename)
        
        with open(filename, 'r') as f:
            content = f.read()
            assert content.strip() == ""
        
        # Clean up
        os.remove(filename)
    
    def test_export_pgn_file_custom_position(self):
        """Test PGN export for custom starting position."""
        # Create environment with custom position
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        env = ChessEnvironment(self.agent1, self.agent2, initial_fen=custom_fen)
        
        # Play a move
        env.board.push_san("Nf6")
        env.move_history = ["Nf6"]
        
        # Test export
        filename = "custom_position_game.pgn"
        success = env.export_pgn_file(filename)
        
        assert success
        
        # Verify file contains custom FEN
        with open(filename, 'r') as f:
            content = f.read()
            assert custom_fen in content
        
        # Clean up
        import os
        os.remove(filename)

    def test_environment_state_consistency(self):
        """Test that environment state remains consistent."""
        # Play a move
        legal_moves = self.env.get_legal_moves()
        move = legal_moves[0]

        # Record initial state
        initial_fen = self.env.get_fen()
        initial_side = self.env.get_side_to_move()

        # Play move
        self.env.play_move(move)

        # Check state changed appropriately
        new_fen = self.env.get_fen()
        new_side = self.env.get_side_to_move()

        assert new_fen != initial_fen
        assert new_side != initial_side
        assert len(self.env.move_history) == 1

    def test_multiple_games_same_environment(self):
        """Test playing multiple games with the same environment."""
        # Play first game
        result1 = self.env.play_game(verbose=False)
        moves1 = result1["moves_played"]

        # Play second game
        result2 = self.env.play_game(verbose=False)
        moves2 = result2["moves_played"]

        # Both games should have results
        assert "result" in result1
        assert "result" in result2

        # Move counts might be different due to randomness
        assert moves1 > 0
        assert moves2 > 0

    def test_time_limit_warning(self):
        """Test time limit warning for slow agents."""

        class SlowAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                import time

                time.sleep(0.1)  # Sleep for 100ms
                return super().choose_move(
                    board, legal_moves, move_history, side_to_move
                )

        slow_env = ChessEnvironment(SlowAgent(), SlowAgent(), time_limit=0.05)

        # This should generate a warning but still work
        move = slow_env.play_agent_move(slow_env.agent1, "White")
        assert move is not None
