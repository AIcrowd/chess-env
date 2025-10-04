"""
Chess agents package.

This package contains implementations of various chess-playing agents.
"""

from .base import ChessAgent
from .first_move_agent import FirstMoveAgent
from .huggingface_agent import HuggingFaceAgent
from .last_move_agent import LastMoveAgent
from .openai_agent import OpenAIAgent
from .random_agent import RandomAgent
from .stockfish_agent import StockfishAgent

__all__ = [
    "ChessAgent",
    "RandomAgent",
    "FirstMoveAgent",
    "LastMoveAgent",
    "StockfishAgent",
    "OpenAIAgent",
    "HuggingFaceAgent",
]
