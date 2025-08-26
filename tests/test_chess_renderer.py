"""
Tests for the chess renderer functionality.
"""

import unittest
from unittest.mock import patch

import chess
from chess_renderer import RICH_AVAILABLE, ChessRenderer


class TestChessRenderer(unittest.TestCase):
    """Test cases for the ChessRenderer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.renderer = ChessRenderer()
        self.board = chess.Board()

    def test_initialization(self):
        """Test renderer initialization."""
        self.assertTrue(self.renderer.show_coordinates)
        self.assertFalse(self.renderer.show_move_numbers)
        self.assertEqual(self.renderer.empty_square_char, "·")
        self.assertEqual(self.renderer.use_rich, RICH_AVAILABLE)

    def test_initialization_custom_options(self):
        """Test renderer initialization with custom options."""
        renderer = ChessRenderer(
            show_coordinates=False,
            show_move_numbers=True,
            empty_square_char=".",
            use_rich=False
        )
        
        self.assertFalse(renderer.show_coordinates)
        self.assertTrue(renderer.show_move_numbers)
        self.assertEqual(renderer.empty_square_char, ".")
        self.assertFalse(renderer.use_rich)

    def test_render_board_basic(self):
        """Test basic board rendering."""
        output = self.renderer.render_board(self.board)
        
        # Should be a string
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        
        # Should contain chess pieces
        self.assertIn("♔", output)  # White king
        self.assertIn("♚", output)  # Black king
        self.assertIn("♙", output)  # White pawns
        self.assertIn("♟", output)  # Black pawns

    def test_render_board_with_coordinates(self):
        """Test board rendering with coordinates."""
        output = self.renderer.render_board(self.board)
        
        # Should contain file coordinates
        self.assertIn("a", output)
        self.assertIn("h", output)
        
        # Should contain rank coordinates
        self.assertIn("1", output)
        self.assertIn("8", output)

    def test_render_board_without_coordinates(self):
        """Test board rendering without coordinates."""
        renderer = ChessRenderer(show_coordinates=False)
        output = renderer.render_board(self.board)
        
        # Should not contain file coordinates
        self.assertNotIn("a", output)
        self.assertNotIn("h", output)
        
        # Should not contain rank coordinates
        self.assertNotIn("1", output)
        self.assertNotIn("8", output)

    def test_render_board_with_move_numbers(self):
        """Test board rendering with move numbers."""
        renderer = ChessRenderer(show_move_numbers=True)
        output = renderer.render_board(self.board, move_number=5)
        
        # Should contain move number
        self.assertIn("Move 5", output)

    def test_render_board_with_last_move_highlight(self):
        """Test board rendering with last move highlighting."""
        # Make a move
        move = chess.Move.from_uci("e2e4")
        self.board.push(move)
        
        output = self.renderer.render_board(self.board, last_move=move)
        
        # Should contain the move
        self.assertIn("♙", output)  # White pawn should be visible

    def test_render_board_empty_square_char(self):
        """Test board rendering with different empty square characters."""
        # Create an empty board
        empty_board = chess.Board()
        empty_board.clear()
        
        # Test different empty square characters
        for char in ["·", ".", "-", " "]:
            renderer = ChessRenderer(empty_square_char=char)
            output = renderer.render_board(empty_board)
            
            # Should contain the empty square character
            self.assertIn(char, output)

    def test_render_game_state(self):
        """Test game state rendering."""
        # Play some moves
        self.board.push(chess.Move.from_uci("e2e4"))
        self.board.push(chess.Move.from_uci("e7e5"))
        move_history = ["e2e4", "e7e5"]
        
        output = self.renderer.render_game_state(
            self.board,
            move_history=move_history,
            side_to_move="White",
            game_result=None
        )
        
        # Should contain game information
        self.assertIn("Side to move: White", output)
        self.assertIn("Moves played: 2", output)
        self.assertIn("Move history:", output)
        self.assertIn("1. e2e4", output)
        self.assertIn("2. e7e5", output)
        
        # Should contain board
        self.assertIn("♙", output)
        self.assertIn("♟", output)

    def test_render_game_state_without_history(self):
        """Test game state rendering without move history."""
        output = self.renderer.render_game_state(
            self.board,
            move_history=None,
            side_to_move="White",
            game_result=None
        )
        
        # Should not contain move history
        self.assertNotIn("Move history:", output)
        
        # Should still contain board and basic info
        self.assertIn("Side to move: White", output)
        self.assertIn("♙", output)

    def test_render_move_sequence(self):
        """Test move sequence rendering."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3")
        ]
    
        output = self.renderer.render_move_sequence(self.board, moves, style="simple")
    
        # Should contain move sequence
        self.assertIn("Move sequence:", output)
        self.assertIn("Move 1: e2e4", output)
        self.assertIn("Move 2: e7e5", output)
        self.assertIn("Move 3: g1f3", output)
    
    def test_render_move_sequence_professional_style(self):
        """Test move sequence rendering with professional style."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5")
        ]
    
        output = self.renderer.render_move_sequence(self.board, moves, style="professional")
    
        # Should contain professional formatting
        self.assertIn("🎯 MOVE SEQUENCE DISPLAY 🎯", output)
        self.assertIn("📝 Move 1: e2e4", output)
        self.assertIn("📝 Move 2: e7e5", output)
        self.assertIn("🏁", output)
    
    def test_render_move_sequence_with_start_fen(self):
        """Test move sequence rendering with custom starting position."""
        moves = [
            chess.Move.from_uci("g8f6"),
            chess.Move.from_uci("d2d4")
        ]
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
        output = self.renderer.render_move_sequence(self.board, moves, start_fen, style="simple")
    
        # Should show initial position
        self.assertIn("Initial position:", output)
        self.assertIn("Move 1: g8f6", output)
        self.assertIn("Move 2: d2d4", output)

    def test_render_position_analysis(self):
        """Test position analysis rendering."""
        output = self.renderer.render_position_analysis(self.board)
        
        # Should contain analysis information
        self.assertIn("Position Analysis:", output)
        self.assertIn("White material: 39", output)
        self.assertIn("Black material: 39", output)
        self.assertIn("Material difference: +0", output)
        self.assertIn("Legal moves: 20", output)
        self.assertIn("Sample moves:", output)
        
        # Should contain board
        self.assertIn("♙", output)
        self.assertIn("♟", output)

    def test_render_position_analysis_with_few_moves(self):
        """Test position analysis rendering with few legal moves."""
        # Create a position with few legal moves
        few_moves_fen = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"  # King and pawn endgame
        board = chess.Board(few_moves_fen)
        
        output = self.renderer.render_position_analysis(board)
        
        # Should contain analysis information
        self.assertIn("Position Analysis:", output)
        self.assertIn("Legal moves:", output)
        
        # Should show sample moves (all moves if <= 10)
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 10:
            for move in legal_moves:
                self.assertIn(move.uci(), output)

    def test_count_material(self):
        """Test material counting."""
        # Starting position should have 39 material for each side
        white_material = self.renderer._count_material(self.board, chess.WHITE)
        black_material = self.renderer._count_material(self.board, chess.BLACK)
        
        self.assertEqual(white_material, 39)
        self.assertEqual(black_material, 39)
        
        # After capturing a pawn, material should decrease
        self.board.push(chess.Move.from_uci("e2e4"))
        self.board.push(chess.Move.from_uci("e7e5"))
        self.board.push(chess.Move.from_uci("e4e5"))  # Capture
        
        white_material_after = self.renderer._count_material(self.board, chess.WHITE)
        black_material_after = self.renderer._count_material(self.board, chess.BLACK)
        
        self.assertEqual(white_material_after, 39)  # White still has all pieces
        self.assertEqual(black_material_after, 38)  # Black lost a pawn

    def test_rich_rendering_availability(self):
        """Test rich rendering availability detection."""
        # Test that rich rendering is properly detected
        if RICH_AVAILABLE:
            self.assertTrue(self.renderer.use_rich)
        else:
            self.assertFalse(self.renderer.use_rich)

    def test_rich_console_initialization(self):
        """Test rich console initialization."""
        if RICH_AVAILABLE:
            self.assertIsNotNone(self.renderer.console)
        else:
            # Should not have console if rich is not available
            self.assertFalse(hasattr(self.renderer, 'console'))

    def test_display_board_rich_method(self):
        """Test the display_board method with rich output."""
        # This method should not raise errors
        try:
            self.renderer.render_board(self.board, output_mode="display")
        except Exception as e:
            self.fail(f"render_board with display mode raised an exception: {e}")
    
    def test_render_board_clean_mode(self):
        """Test the render_board method with clean output mode."""
        # Clean mode should return empty string when rich is available
        output = self.renderer.render_board(self.board, output_mode="clean")
        if self.renderer.use_rich:
            self.assertEqual(output, "")
        else:
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 0)

    def test_renderer_options_consistency(self):
        """Test that renderer options are consistently applied."""
        # Test setting various options
        test_options = [
            (True, False, "·", True),
            (False, True, ".", False),
            (True, True, "-", True),
            (False, False, " ", False),
        ]
        
        for show_coords, show_numbers, empty_char, use_rich in test_options:
            with self.subTest(options=test_options):
                renderer = ChessRenderer(
                    show_coordinates=show_coords,
                    show_move_numbers=show_numbers,
                    empty_square_char=empty_char,
                    use_rich=use_rich
                )
                
                # Verify options were set correctly
                self.assertEqual(renderer.show_coordinates, show_coords)
                self.assertEqual(renderer.show_move_numbers, show_numbers)
                self.assertEqual(renderer.empty_square_char, empty_char)
                self.assertEqual(renderer.use_rich, use_rich and RICH_AVAILABLE)

    def test_board_coordinates_consistency(self):
        """Test that board coordinates are consistent."""
        # Files should be a-h
        expected_files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.assertEqual(self.renderer.FILES, expected_files)
        
        # Ranks should be 8-1 (top to bottom)
        expected_ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        self.assertEqual(self.renderer.RANKS, expected_ranks)

    def test_piece_symbols_consistency(self):
        """Test that piece symbols are consistent."""
        # Test that all piece symbols are defined
        expected_pieces = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
        
        for piece in expected_pieces:
            self.assertIn(piece, self.renderer.PIECES)
            self.assertIsInstance(self.renderer.PIECES[piece], str)
            self.assertEqual(len(self.renderer.PIECES[piece]), 1)  # Single character

    def test_empty_board_rendering(self):
        """Test rendering of an empty board."""
        empty_board = chess.Board()
        empty_board.clear()
        
        output = self.renderer.render_board(empty_board)
        
        # Should contain empty square characters
        self.assertIn(self.renderer.empty_square_char, output)
        
        # Should not contain any piece symbols
        for piece_symbol in self.renderer.PIECES.values():
            self.assertNotIn(piece_symbol, output)

    def test_single_piece_rendering(self):
        """Test rendering of a board with a single piece."""
        # Place just a king on the board
        single_piece_board = chess.Board()
        single_piece_board.clear()
        single_piece_board.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.WHITE))
        
        output = self.renderer.render_board(single_piece_board)
        
        # Should contain the king symbol
        self.assertIn("♔", output)
        
        # Should contain empty square characters
        self.assertIn(self.renderer.empty_square_char, output)

    def test_move_highlighting_edge_cases(self):
        """Test move highlighting with edge cases."""
        # Test with no last move
        output = self.renderer.render_board(self.board, last_move=None)
        self.assertIsInstance(output, str)
        
        # Test with invalid move (should not crash)
        invalid_move = chess.Move(chess.E1, chess.E1)  # Same square
        output = self.renderer.render_board(self.board, last_move=invalid_move)
        self.assertIsInstance(output, str)

    def test_coordinate_parsing(self):
        """Test that coordinate parsing works correctly."""
        # Test that we can parse all valid coordinates
        for file in self.renderer.FILES:
            for rank in self.renderer.RANKS:
                square_str = file + rank
                try:
                    square = chess.parse_square(square_str)
                    self.assertIsInstance(square, int)
                    self.assertGreaterEqual(square, 0)
                    self.assertLessEqual(square, 63)
                except ValueError:
                    self.fail(f"Failed to parse coordinate: {square_str}")


if __name__ == "__main__":
    unittest.main()
