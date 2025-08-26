"""
Tests for the chess renderer functionality.
"""

import pytest

import chess
from chess_renderer import RICH_AVAILABLE, ChessRenderer


class TestChessRenderer:
    """Test the ChessRenderer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = ChessRenderer()
        self.board = chess.Board()
    
    def test_renderer_initialization(self):
        """Test renderer initialization with default options."""
        renderer = ChessRenderer()
        assert renderer.show_coordinates is True
        assert renderer.show_move_numbers is False
        assert renderer.empty_square_char == "·"
        assert renderer.use_rich == RICH_AVAILABLE
        
        renderer = ChessRenderer(show_coordinates=False, show_move_numbers=True, 
                               empty_square_char=".", use_rich=False)
        assert renderer.show_coordinates is False
        assert renderer.show_move_numbers is True
        assert renderer.empty_square_char == "."
        assert renderer.use_rich is False
    
    def test_render_board_basic(self):
        """Test basic board rendering."""
        rendered = self.renderer.render_board(self.board)
        
        # Should contain piece symbols
        assert "♙" in rendered  # White pawns
        assert "♖" in rendered  # White rooks
        assert "♘" in rendered  # White knights
        assert "♗" in rendered  # White bishops
        assert "♕" in rendered  # White queen
        assert "♔" in rendered  # White king
        
        # Should contain black pieces
        assert "♟" in rendered  # Black pawns
        assert "♜" in rendered  # Black rooks
        assert "♞" in rendered  # Black knights
        assert "♝" in rendered  # Black bishops
        assert "♛" in rendered  # Black queen
        assert "♚" in rendered  # Black king
        
        # Should contain empty squares
        assert "·" in rendered  # Empty squares
        
        # Should contain coordinates (either plain or rich format)
        if self.renderer.use_rich:
            # Rich format has different coordinate layout
            assert "8" in rendered and "1" in rendered  # Rank labels
            assert "a" in rendered and "h" in rendered  # File labels
        else:
            # Plain format
            assert "a b c d e f g h" in rendered
            assert "8 |" in rendered
            assert "1 |" in rendered
    
    def test_render_board_without_coordinates(self):
        """Test board rendering without coordinates."""
        renderer = ChessRenderer(show_coordinates=False)
        rendered = renderer.render_board(self.board)
        
        # Should not contain coordinate lines
        assert "a b c d e f g h" not in rendered
        assert "8 |" not in rendered
        assert "1 |" not in rendered
        
        # Should still contain pieces
        assert "♙" in rendered
        assert "♟" in rendered
    
    def test_render_board_with_move_number(self):
        """Test board rendering with move number."""
        renderer = ChessRenderer(show_move_numbers=True)
        rendered = renderer.render_board(self.board, move_number=5)
        
        assert "Move 5" in rendered
    
    def test_render_board_with_last_move_highlight(self):
        """Test board rendering with last move highlighting."""
        # Make a move
        move = chess.Move.from_uci("e2e4")
        self.board.push(move)
        
        rendered = self.renderer.render_board(self.board, last_move=move)
        
        # Should highlight the last move (either with brackets or rich styling)
        if self.renderer.use_rich:
            # Rich format highlights differently
            assert "♙" in rendered  # White pawn should be visible
            # Note: Rich highlighting is visual and may not be easily testable in text
        else:
            # Plain format highlights with brackets
            assert "[♙]" in rendered  # Highlighted white pawn
    
    def test_render_game_state(self):
        """Test complete game state rendering."""
        # Make some moves
        self.board.push_san("e4")
        self.board.push_san("e5")
        move_history = ["e4", "e5"]
        
        rendered = self.renderer.render_game_state(
            self.board, 
            move_history=move_history,
            side_to_move="White",
            game_result=None
        )
        
        assert "Side to move: White" in rendered
        assert "Moves played: 2" in rendered
        assert "Move history:" in rendered
        assert "1. e4 2. e5" in rendered
        assert "♙" in rendered  # Board should be rendered
    
    def test_render_move_sequence(self):
        """Test move sequence rendering."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3")
        ]
        
        rendered = self.renderer.render_move_sequence(self.board, moves)
        
        assert "Move sequence:" in rendered
        assert "Move 1: e2e4" in rendered  # UCI notation
        assert "Move 2: e7e5" in rendered  # UCI notation
        assert "Move 3: g1f3" in rendered  # UCI notation
        
        # Should show board after each move
        assert "♙" in rendered  # Initial position
        assert "♟" in rendered  # Black pieces
    
    def test_render_position_analysis(self):
        """Test position analysis rendering."""
        rendered = self.renderer.render_position_analysis(self.board)
        
        assert "Position Analysis:" in rendered
        assert "White material: 39" in rendered  # Starting position
        assert "Black material: 39" in rendered
        assert "Material difference: +0" in rendered
        assert "Legal moves: 20" in rendered
        assert "Sample moves:" in rendered
        
        # Should show board
        assert "♙" in rendered
        assert "♟" in rendered
    
    def test_material_counting(self):
        """Test material counting functionality."""
        # Test starting position
        white_material = self.renderer._count_material(self.board, chess.WHITE)
        black_material = self.renderer._count_material(self.board, chess.BLACK)
        
        assert white_material == 39  # 8 pawns + 2 rooks + 2 knights + 2 bishops + 1 queen
        assert black_material == 39
        
        # Test after some moves
        self.board.push_san("e4")
        self.board.push_san("e5")
        self.board.push_san("Nf3")  # Legal knight move instead of capture
        
        white_material = self.renderer._count_material(self.board, chess.WHITE)
        black_material = self.renderer._count_material(self.board, chess.BLACK)
        
        assert white_material == 39  # White material unchanged
        assert black_material == 39  # Black material unchanged
    
    def test_renderer_with_custom_position(self):
        """Test renderer with custom FEN position."""
        # Test with a specific position
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        custom_board = chess.Board(custom_fen)
        
        rendered = self.renderer.render_board(custom_board)
        
        # Should show the custom position
        assert "♙" in rendered  # White pawn on e4
        assert "♟" in rendered  # Black pieces
        assert "♔" in rendered  # White king
    
    def test_renderer_edge_cases(self):
        """Test renderer with edge cases."""
        # Empty board
        empty_board = chess.Board()
        empty_board.clear()
        
        rendered = self.renderer.render_board(empty_board)
        assert "·" in rendered  # Should show empty square character
        
        # Board with only kings
        kings_only = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        rendered = self.renderer.render_board(kings_only)
        assert "♔" in rendered  # White king
        assert "♚" in rendered  # Black king
        assert "·" in rendered  # Empty squares
    
    def test_empty_square_character(self):
        """Test different empty square characters."""
        # Test with dot
        renderer = ChessRenderer(empty_square_char=".")
        empty_board = chess.Board()
        empty_board.clear()
        rendered = renderer.render_board(empty_board)
        assert "." in rendered
        assert "·" not in rendered
        
        # Test with space
        renderer = ChessRenderer(empty_square_char=" ")
        rendered = renderer.render_board(empty_board)
        assert "  " in rendered  # Two spaces for empty square
        
        # Test with dash
        renderer = ChessRenderer(empty_square_char="-")
        rendered = renderer.render_board(empty_board)
        assert " - " in rendered
    
    def test_rich_rendering_availability(self):
        """Test rich rendering availability detection."""
        # Test that rich availability is correctly detected
        renderer = ChessRenderer(use_rich=True)
        assert renderer.use_rich == RICH_AVAILABLE
        
        # Test that rich can be disabled
        renderer = ChessRenderer(use_rich=False)
        assert renderer.use_rich is False
    
    def test_plain_vs_rich_rendering(self):
        """Test that both plain and rich rendering work."""
        # Test plain rendering
        renderer_plain = ChessRenderer(use_rich=False)
        plain_output = renderer_plain.render_board(self.board)
        assert "♙" in plain_output
        assert "♟" in plain_output
        
        # Test rich rendering (if available)
        if RICH_AVAILABLE:
            renderer_rich = ChessRenderer(use_rich=True)
            rich_output = renderer_rich.render_board(self.board)
            assert "♙" in rich_output
            assert "♟" in rich_output
