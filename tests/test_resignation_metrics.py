import unittest

import chess

from agents import ChessAgent
from env import ChessEnvironment
from metrics import apply_resignation_cpl_adjustment


class ResigningAgent(ChessAgent):
    """Agent that resigns immediately."""

    def choose_move(self, board, legal_moves, move_history, side_to_move):
        return None, "resign"


class FirstLegalAgent(ChessAgent):
    """Agent that always plays the first legal move."""

    def choose_move(self, board, legal_moves, move_history, side_to_move):
        return legal_moves[0], None


class TestResignationMetrics(unittest.TestCase):
    def test_env_records_white_resignation(self):
        env = ChessEnvironment(ResigningAgent(), FirstLegalAgent())
        result = env.play_game(verbose=False)

        self.assertEqual(result["game_over_reason"], "resignation")
        self.assertEqual(result["resigned_side"], "White")
        self.assertEqual(len(result["move_history"]), 0)

    def test_env_records_black_resignation(self):
        env = ChessEnvironment(FirstLegalAgent(), ResigningAgent())
        result = env.play_game(verbose=False)

        self.assertEqual(result["game_over_reason"], "resignation")
        self.assertEqual(result["resigned_side"], "Black")
        # White should have made exactly one move
        self.assertEqual(len(result["move_history"]), 1)

    def test_apply_adjustment_immediate_white_resign_sets_black_zero(self):
        analysis = {"white_acpl": 1000.0, "black_acpl": 1000.0}
        game_stats = {
            "game_over_reason": "resignation",
            "resigned_side": "White",
            "move_history": [],
            "starting_turn_white": True,
        }

        updated = apply_resignation_cpl_adjustment(analysis, game_stats, penalty=1000.0)
        self.assertEqual(updated["white_acpl"], 1000.0)
        self.assertEqual(updated["black_acpl"], 0.0)

    def test_apply_adjustment_black_resigns_after_one_move(self):
        # White made one move; Black resigns; treat resignation as Black's first move with CPL=1000
        analysis = {"white_acpl": 12.3, "black_acpl": 45.6}
        game_stats = {
            "game_over_reason": "resignation",
            "resigned_side": "Black",
            "move_history": ["e2e4"],
            "starting_turn_white": True,
        }

        updated = apply_resignation_cpl_adjustment(analysis, game_stats, penalty=1000.0)
        self.assertEqual(updated["white_acpl"], 12.3)
        # black_moves=0 -> sum=0, after resignation moves=1 sum=1000 => acpl=1000
        self.assertEqual(updated["black_acpl"], 1000.0)

    def test_apply_adjustment_no_resignation_no_change(self):
        analysis = {"white_acpl": 5.0, "black_acpl": 7.0}
        game_stats = {
            "game_over_reason": "stalemate",
            "resigned_side": None,
            "move_history": ["e2e4", "e7e5"],
            "starting_turn_white": True,
        }

        updated = apply_resignation_cpl_adjustment(analysis, game_stats, penalty=1000.0)
        self.assertEqual(updated, analysis)

    def test_apply_adjustment_white_resigns_after_two_moves(self):
        # White CPLs: 20, 60; Black CPL: 10; then White resigns => ACPL = (20+60+1000)/3 = 360
        analysis = {"white_acpl": 40.0, "black_acpl": 10.0}
        game_stats = {
            "game_over_reason": "resignation",
            "resigned_side": "White",
            "move_history": ["e2e4", "e7e5", "a2a4"],
            "starting_turn_white": True,
        }

        updated = apply_resignation_cpl_adjustment(analysis, game_stats, penalty=1000.0)
        self.assertAlmostEqual(updated["white_acpl"], (20 + 60 + 1000) / 3, places=3)
        self.assertEqual(updated["black_acpl"], 10.0)


if __name__ == "__main__":
    unittest.main()
