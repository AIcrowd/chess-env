"""Small helpers for post-game metric adjustments."""

from __future__ import annotations

from typing import Dict, Any


def _count_moves_by_side(move_history_len: int, starting_turn_white: bool) -> tuple[int, int]:
    """Return (white_moves, black_moves) given plies and starting side."""
    if move_history_len <= 0:
        return (0, 0)
    if starting_turn_white:
        white_moves = (move_history_len + 1) // 2
        black_moves = move_history_len // 2
    else:
        black_moves = (move_history_len + 1) // 2
        white_moves = move_history_len // 2
    return white_moves, black_moves


def apply_resignation_cpl_adjustment(
    analysis: Dict[str, Any],
    game_stats: Dict[str, Any],
    penalty: float = 1000.0,
) -> Dict[str, Any]:
    """Model resignation as one extra move with CPL=penalty for the resigning side.

    New ACPL (resigner) = (old_sum + penalty) / (old_moves + 1)
    Non-resigner keeps existing ACPL, except if they made 0 moves → force 0.0
    """

    if game_stats.get("game_over_reason") != "resignation":
        return analysis

    side = game_stats.get("resigned_side")
    moves = game_stats.get("move_history", []) or []
    starting_turn_white = game_stats.get("starting_turn_white", True)

    white_moves, black_moves = _count_moves_by_side(len(moves), starting_turn_white)

    updated = dict(analysis)

    # Reconstruct sums from averages and counts (approach B)
    white_sum = updated.get("white_acpl", 0.0) * white_moves
    black_sum = updated.get("black_acpl", 0.0) * black_moves

    if side == "White":
        white_moves += 1
        white_sum += penalty
        updated["white_acpl"] = white_sum / white_moves
        if black_moves == 0:
            updated["black_acpl"] = 0.0
    elif side == "Black":
        black_moves += 1
        black_sum += penalty
        updated["black_acpl"] = black_sum / black_moves
        if white_moves == 0:
            updated["white_acpl"] = 0.0
    else:
        return analysis

    return updated
