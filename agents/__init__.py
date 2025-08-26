"""
Chess agents package.

This package contains implementations of various chess-playing agents.
"""

from .base import ChessAgent
from .first_move_agent import FirstMoveAgent
from .last_move_agent import LastMoveAgent
from .random_agent import RandomAgent

__all__ = ["ChessAgent", "RandomAgent", "FirstMoveAgent", "LastMoveAgent"]
