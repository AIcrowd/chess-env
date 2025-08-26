import pytest

import chess
from agents import ChessAgent, RandomAgent
from env import ChessEnvironment


class TestIntegration:
    """Integration tests for the complete chess environment workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent1 = RandomAgent()
        self.agent2 = RandomAgent()
        self.env = ChessEnvironment(self.agent1, self.agent2, max_moves=50)

    def test_complete_game_workflow(self):
        """Test a complete game from start to finish."""
        # Start with fresh environment
        assert self.env.get_fen() == chess.STARTING_FEN
        assert self.env.get_side_to_move() == "White"
        assert len(self.env.get_legal_moves()) == 20

        # Play a complete game
        result = self.env.play_game(verbose=False)

        # Verify game completed
        assert result["result"] is not None
        assert result["moves_played"] > 0
        assert result["moves_played"] <= 50  # Should not exceed max_moves

        # Verify final state
        assert self.env.is_game_over() or result["moves_played"] == 50
        assert len(self.env.move_history) == result["moves_played"]

    def test_multiple_games_workflow(self):
        """Test playing multiple games with the same environment."""
        results = []

        for i in range(3):
            result = self.env.play_game(verbose=False)
            results.append(result)

            # Verify each game has proper structure
            assert "result" in result
            assert "moves_played" in result
            assert "move_history" in result

            # Verify move history consistency
            assert len(result["move_history"]) == result["moves_played"]

        # With random agents, games might have similar lengths, so we just verify they complete
        move_counts = [r["moves_played"] for r in results]
        assert all(count > 0 for count in move_counts)
        assert all(count <= 50 for count in move_counts)

    def test_agent_interface_consistency(self):
        """Test that agent interface is consistent throughout the game."""

        # Create a custom agent that tracks calls
        class TrackingAgent(RandomAgent):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.received_boards = []
                self.received_moves = []
                self.received_history = []
                self.received_sides = []

            def choose_move(self, board, legal_moves, move_history, side_to_move):
                self.call_count += 1
                self.received_boards.append(board.fen())
                self.received_moves.append(len(legal_moves))
                self.received_history.append(move_history.copy())
                self.received_sides.append(side_to_move)
                return super().choose_move(
                    board, legal_moves, move_history, side_to_move
                )

        tracking_agent = TrackingAgent()
        env = ChessEnvironment(tracking_agent, self.agent2, max_moves=20)

        # Play a game
        result = env.play_game(verbose=False)

        # Verify agent was called appropriately
        assert tracking_agent.call_count > 0
        assert len(tracking_agent.received_boards) == tracking_agent.call_count
        assert len(tracking_agent.received_moves) == tracking_agent.call_count
        assert len(tracking_agent.received_history) == tracking_agent.call_count
        assert len(tracking_agent.received_sides) == tracking_agent.call_count

        # Verify side alternation
        white_calls = [
            side for side in tracking_agent.received_sides if side == "White"
        ]
        black_calls = [
            side for side in tracking_agent.received_sides if side == "Black"
        ]
        assert len(white_calls) >= len(
            black_calls
        )  # White starts, so should have at least as many calls

    def test_board_state_consistency(self):
        """Test that board state remains consistent throughout the game."""
        # Track board states
        board_states = []
        move_counts = []

        # Play a few moves manually to test consistency
        for i in range(5):
            board_states.append(self.env.get_fen())
            move_counts.append(len(self.env.get_legal_moves()))

            # Play a move
            legal_moves = self.env.get_legal_moves()
            if legal_moves:
                move = legal_moves[0]
                self.env.play_move(move)

        # Verify board states changed
        assert len(set(board_states)) > 1

        # Verify move counts are reasonable
        assert all(count > 0 for count in move_counts)
        assert all(
            count <= 218 for count in move_counts
        )  # Maximum possible legal moves in chess

    def test_error_handling_integration(self):
        """Test error handling in the complete workflow."""

        # Create an agent that sometimes fails
        class UnreliableAgent(RandomAgent):
            def __init__(self, failure_rate=0.3):
                super().__init__()
                self.failure_rate = failure_rate
                self.call_count = 0

            def choose_move(self, board, legal_moves, move_history, side_to_move):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    raise Exception("Simulated failure")
                return super().choose_move(
                    board, legal_moves, move_history, side_to_move
                )

        unreliable_agent = UnreliableAgent()
        env = ChessEnvironment(unreliable_agent, self.agent2, max_moves=10)

        # This should handle errors gracefully
        result = env.play_game(verbose=False)

        # Game should still complete
        assert "result" in result
        assert result["moves_played"] > 0

    def test_pgn_generation_integration(self):
        """Test PGN generation with actual game data."""
        # Play a game
        result = self.env.play_game(verbose=False)

        # Generate PGN
        pgn = self.env.get_pgn()

        # Verify PGN structure
        assert '[Event "Chess Game"]' in pgn
        assert '[White "RandomAgent"]' in pgn
        assert '[Black "RandomAgent"]' in pgn

        # Verify moves are included
        if result["moves_played"] > 0:
            assert len(pgn.split("\n")) > 4  # Header + moves
            # Check that some moves are present
            move_line = pgn.split("\n")[-1]
            assert len(move_line.strip()) > 0

    def test_custom_starting_position_integration(self):
        """Test playing games from custom starting positions."""
        # Test with a midgame position
        midgame_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        self.env.reset(midgame_fen)

        # Verify position is set correctly (accounting for FEN normalization)
        board = chess.Board(midgame_fen)
        assert self.env.get_fen() == board.fen()
        assert self.env.get_side_to_move() == "Black"

        # Play a game from this position
        result = self.env.play_game(verbose=False)

        # Verify game completed
        assert result["result"] is not None
        assert result["moves_played"] > 0

    def test_time_limit_integration(self):
        """Test time limit enforcement in complete games."""

        # Create a slow agent
        class SlowAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                import time

                time.sleep(0.05)  # 50ms delay
                return super().choose_move(
                    board, legal_moves, move_history, side_to_move
                )

        slow_agent = SlowAgent()
        env = ChessEnvironment(slow_agent, slow_agent, time_limit=0.1, max_moves=10)

        # Play a game - should complete despite being slow
        result = env.play_game(verbose=False)

        # Verify game completed
        assert result["result"] is not None
        assert result["moves_played"] > 0

    def test_agent_switching_integration(self):
        """Test switching agents between games."""

        # Create different types of agents
        class FirstMoveAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                # Always choose the first legal move
                return legal_moves[0]

        class LastMoveAgent(RandomAgent):
            def choose_move(self, board, legal_moves, move_history, side_to_move):
                # Always choose the last legal move
                return legal_moves[-1]

        first_agent = FirstMoveAgent()
        last_agent = LastMoveAgent()

        # Play game with first agent vs random
        env1 = ChessEnvironment(first_agent, self.agent2, max_moves=10)
        result1 = env1.play_game(verbose=False)

        # Play game with last agent vs random
        env2 = ChessEnvironment(last_agent, self.agent2, max_moves=10)
        result2 = env2.play_game(verbose=False)

        # Both games should complete
        assert result1["result"] is not None
        assert result2["result"] is not None

        # Games might have different outcomes due to different strategies
        assert result1["moves_played"] > 0
        assert result2["moves_played"] > 0

    def test_edge_case_integration(self):
        """Test edge cases in the complete workflow."""
        # Test with very small move limits
        env = ChessEnvironment(self.agent1, self.agent2, max_moves=1)
        result = env.play_game(verbose=False)

        assert result["game_over_reason"] == "max_moves"
        assert result["moves_played"] == 1

        # Test with very large move limits
        env = ChessEnvironment(self.agent1, self.agent2, max_moves=1000)
        result = env.play_game(verbose=False)

        # Game should complete naturally before reaching 1000 moves
        assert result["result"] is not None
        assert result["moves_played"] < 1000  # Should finish naturally

    def test_performance_integration(self):
        """Test performance characteristics of the complete system."""
        import time

        # Measure time for a complete game
        start_time = time.time()
        result = self.env.play_game(verbose=False)
        end_time = time.time()

        game_time = end_time - start_time

        # Game should complete in reasonable time
        assert game_time < 10.0  # Should complete in under 10 seconds

        # Verify game completed successfully
        assert result["result"] is not None
        assert result["moves_played"] > 0

        # Performance should scale reasonably with move count
        moves_per_second = result["moves_played"] / game_time
        assert moves_per_second > 1.0  # Should process at least 1 move per second
