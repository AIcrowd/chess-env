"""
Chess board renderer for terminal display using Unicode chess pieces.
"""

from typing import List, Optional, Union

import chess

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ChessRenderer:
    """Renders chess boards to the terminal using Unicode characters."""
    
    # Unicode chess piece characters
    PIECES = {
        'P': '♙',  # White pawn
        'R': '♖',  # White rook
        'N': '♘',  # White knight
        'B': '♗',  # White bishop
        'Q': '♕',  # White queen
        'K': '♔',  # White king
        
        'p': '♟',  # Black pawn
        'r': '♜',  # Black rook
        'n': '♞',  # Black knight
        'b': '♝',  # Black bishop
        'q': '♛',  # Black queen
        'k': '♚',  # Black king
    }
    
    # Board coordinates
    FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    def __init__(self, show_coordinates: bool = True, show_move_numbers: bool = False, 
                 empty_square_char: str = "·", use_rich: bool = True):
        """
        Initialize the chess renderer.
        
        Args:
            show_coordinates: Whether to show file/rank coordinates
            show_move_numbers: Whether to show move numbers in the display
            empty_square_char: Character to show for empty squares
            use_rich: Whether to use rich CLI for enhanced rendering (if available)
        """
        self.show_coordinates = show_coordinates
        self.show_move_numbers = show_move_numbers
        self.empty_square_char = empty_square_char
        self.use_rich = use_rich and RICH_AVAILABLE
        
        # Initialize rich console if available
        if self.use_rich:
            self.console = Console(force_terminal=True)
    
    def render_board(self, board: chess.Board, 
                    last_move: Optional[chess.Move] = None,
                    move_number: Optional[int] = None,
                    output_mode: str = "auto") -> Union[str, None]:
        """
        Render a chess board.
        
        Args:
            board: The chess board to render
            last_move: Optional last move to highlight
            move_number: Optional move number to display
            output_mode: "auto" (smart choice), "string" (return string), 
                        "display" (display to terminal), "clean" (no duplicates)
            
        Returns:
            String representation if output_mode is "string", None otherwise
        """
        if output_mode == "string":
            return self._render_board_plain(board, last_move, move_number)
        elif output_mode == "display":
            if self.use_rich:
                self._display_board_rich(board, last_move, move_number)
            else:
                print(self._render_board_plain(board, last_move, move_number))
            return None
        elif output_mode == "clean":
            if self.use_rich:
                self._display_board_rich(board, last_move, move_number)
                return ""  # Return empty string to avoid duplication
            else:
                return self._render_board_plain(board, last_move, move_number)
        else:  # "auto" mode
            if self.use_rich:
                self._display_board_rich(board, last_move, move_number)
                return self._render_board_plain(board, last_move, move_number)
            else:
                return self._render_board_plain(board, last_move, move_number)
    
    def display_board(self, board: chess.Board, 
                     last_move: Optional[chess.Move] = None,
                     move_number: Optional[int] = None) -> None:
        """
        Display board to terminal (convenience method).
        
        Args:
            board: The chess board to render
            last_move: Optional last move to highlight
            move_number: Optional move number to display
        """
        self.render_board(board, last_move, move_number, output_mode="display")
    
    def _display_board_rich(self, board: chess.Board, 
                           last_move: Optional[chess.Move] = None,
                           move_number: Optional[int] = None) -> None:
        """Display board using rich CLI directly to terminal."""
        # Create a table for the chess board
        table = Table(show_header=False, show_edge=True, box=None)
        
        # Add columns for files
        for file in self.FILES:
            table.add_column(file, justify="center", width=3)
        
        # Add move number if requested
        if self.show_move_numbers and move_number is not None:
            move_text = Text(f"Move {move_number}", style="bold blue")
            table.title = move_text
        
        # Render board squares
        for rank_idx, rank in enumerate(self.RANKS):
            row_cells = []
            
            for file_idx, file in enumerate(self.FILES):
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                
                # Determine background color (alternating light/dark)
                is_light_square = (file_idx + rank_idx) % 2 == 0
                
                # Check if this is the last move
                is_last_move = (last_move is not None and 
                              (square == last_move.from_square or 
                               square == last_move.to_square))
                
                # Get piece symbol or empty square character
                if piece is None:
                    piece_char = self.empty_square_char
                    # Empty squares use contrasting colors
                    if is_light_square:
                        text = Text(f" {piece_char} ", style="black on white")
                    else:
                        text = Text(f" {piece_char} ", style="white on black")
                else:
                    piece_char = self.PIECES[piece.symbol()]
                    
                    # Pieces use high contrast colors on all backgrounds
                    if piece.color == chess.WHITE:
                        # White pieces - always high contrast
                        if is_light_square:
                            text = Text(f" {piece_char} ", style="black on white")
                        else:
                            text = Text(f" {piece_char} ", style="white on black")
                    else:
                        # Black pieces - always high contrast
                        if is_light_square:
                            text = Text(f" {piece_char} ", style="black on white")
                        else:
                            text = Text(f" {piece_char} ", style="white on black")
                
                # Highlight last move if needed
                if is_last_move:
                    text = Text(f" {piece_char} ", style="bold on yellow")
                
                row_cells.append(text)
            
            # Add row with rank label
            if self.show_coordinates:
                rank_label = Text(f" {rank} ", style="bold cyan")
                row_cells.insert(0, rank_label)
                table.add_row(*row_cells)
            else:
                table.add_row(*row_cells)
        
        # Add coordinate header if requested
        if self.show_coordinates:
            header_cells = [Text("   ", style="bold cyan")]  # Empty cell for rank column
            for file in self.FILES:
                header_cells.append(Text(f" {file} ", style="bold cyan"))
            table.add_row(*header_cells, style="bold cyan")
        
        # Display directly to terminal
        self.console.print(table)
    
    def _render_board_plain(self, board: chess.Board, 
                           last_move: Optional[chess.Move] = None,
                           move_number: Optional[int] = None) -> str:
        """Render board using plain text."""
        lines = []
        
        # Add move number if requested
        if self.show_move_numbers and move_number is not None:
            lines.append(f"Move {move_number}")
            lines.append("")
        
        # Add top coordinate line
        if self.show_coordinates:
            coord_line = "   " + " ".join(self.FILES) + "  "
            lines.append(coord_line)
            lines.append("  +" + "-" * 15 + "+")
        
        # Render board squares
        for rank_idx, rank in enumerate(self.RANKS):
            line_parts = []
            
            # Add rank coordinate
            if self.show_coordinates:
                line_parts.append(f"{rank} |")
            
            # Add squares
            for file_idx, file in enumerate(self.FILES):
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                
                # Determine background color (alternating light/dark)
                is_light_square = (file_idx + rank_idx) % 2 == 0
                
                # Check if this is the last move
                is_last_move = (last_move is not None and 
                              (square == last_move.from_square or 
                               square == last_move.to_square))
                
                # Get piece symbol or empty square character
                if piece is None:
                    piece_char = self.empty_square_char
                else:
                    piece_char = self.PIECES[piece.symbol()]
                
                # Format square with appropriate highlighting
                if is_last_move:
                    # Highlight last move with brackets
                    square_str = f"[{piece_char}]"
                else:
                    # Normal square
                    square_str = f" {piece_char} "
                
                line_parts.append(square_str)
            
            # Add closing coordinate
            if self.show_coordinates:
                line_parts.append(f"| {rank}")
            
            lines.append("".join(line_parts))
        
        # Add bottom coordinate line
        if self.show_coordinates:
            lines.append("  +" + "-" * 15 + "+")
            coord_line = "   " + " ".join(self.FILES) + "  "
            lines.append(coord_line)
        
        return "\n".join(lines)
    
    def render_game_state(self, board: chess.Board, 
                         move_history: List[str] = None,
                         side_to_move: str = None,
                         game_result: str = None) -> str:
        """
        Render a complete game state including board and information.
        
        Args:
            board: The chess board to render
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move
            game_result: Current game result if game is over
            
        Returns:
            String representation of the complete game state
        """
        lines = []
        
        # Add game information
        if side_to_move:
            lines.append(f"Side to move: {side_to_move}")
        
        if game_result:
            lines.append(f"Game result: {game_result}")
        
        if move_history:
            lines.append(f"Moves played: {len(move_history)}")
        
        if any([side_to_move, game_result, move_history]):
            lines.append("")
        
        # Add board
        lines.append(self.render_board(board, output_mode="string"))
        
        # Add move history if provided
        if move_history:
            lines.append("")
            lines.append("Move history:")
            # Format moves in groups of 10 for readability
            for i in range(0, len(move_history), 10):
                move_group = move_history[i:i+10]
                move_numbers = [f"{i+j+1}." for j in range(len(move_group))]
                moves_with_numbers = [f"{num} {move}" for num, move in zip(move_numbers, move_group)]
                lines.append(" ".join(moves_with_numbers))
        
        return "\n".join(lines)
    
    def render_move_sequence(self, board: chess.Board, 
                           moves: List[chess.Move],
                           start_fen: str = None,
                           style: str = "professional",
                           spacing: int = 2) -> str:
        """
        Render a sequence of moves showing the board after each move.
        
        Args:
            board: The chess board to render
            moves: List of moves to show
            start_fen: Optional starting FEN position
            style: "simple", "professional", or "fancy"
            spacing: Number of empty lines between boards (1-5)
            
        Returns:
            String representation of the move sequence
        """
        lines = []
        
        # Create a temporary board for move sequence
        temp_board = chess.Board(start_fen) if start_fen else chess.Board()
        
        # Apply spacing constraints
        spacing = max(1, min(5, spacing))
        
        # Choose style
        if style == "simple":
            lines.append("Move sequence:")
            lines.append("")
            
            # Show initial position
            if start_fen:
                lines.append("Initial position:")
                lines.append(self._render_board_plain(temp_board))
                lines.append("")
            
            # Show board after each move
            for i, move in enumerate(moves):
                temp_board.push(move)
                uci_move = move.uci()
                
                lines.append("=" * 60)
                lines.append(f"Move {i+1}: {uci_move}")
                lines.append("=" * 60)
                lines.append("")
                
                lines.append(self._render_board_plain(temp_board, last_move=move))
                lines.append("")
                
                if i < len(moves) - 1:
                    for _ in range(spacing):
                        lines.append("")
        
        elif style == "professional":
            lines.append("🎯 MOVE SEQUENCE DISPLAY 🎯")
            lines.append("")
            
            # Show initial position
            if start_fen:
                lines.append("📍 Initial Position:")
                lines.append(self._render_board_plain(temp_board))
                lines.append("")
            
            # Show board after each move
            for i, move in enumerate(moves):
                temp_board.push(move)
                uci_move = move.uci()
                
                lines.append("🚀" + "=" * 56 + "🚀")
                lines.append(f"📝 Move {i+1}: {uci_move}")
                lines.append("🚀" + "=" * 56 + "🚀")
                lines.append("")
                
                lines.append(self._render_board_plain(temp_board, last_move=move))
                lines.append("")
                
                if i < len(moves) - 1:
                    for _ in range(spacing):
                        lines.append("")
            
            lines.append("🏁" + "=" * 56 + "🏁")
        
        elif style == "fancy":
            lines.append("✨ ✨ ✨  MOVE SEQUENCE DISPLAY  ✨ ✨ ✨")
            lines.append("")
            
            # Show initial position
            if start_fen:
                lines.append("🌟 Initial Position:")
                lines.append(self._render_board_plain(temp_board))
                lines.append("")
            
            # Show board after each move
            for i, move in enumerate(moves):
                temp_board.push(move)
                uci_move = move.uci()
                
                lines.append("💫" + "~" * 58 + "💫")
                lines.append(f"🎮 Move {i+1}: {uci_move}")
                lines.append("💫" + "~" * 58 + "💫")
                lines.append("")
                
                lines.append(self._render_board_plain(temp_board, last_move=move))
                lines.append("")
                
                if i < len(moves) - 1:
                    for _ in range(spacing):
                        lines.append("")
            
            lines.append("🎉" + "~" * 58 + "🎉")
        
        return "\n".join(lines)
    
    def render_position_analysis(self, board: chess.Board) -> str:
        """
        Render position analysis including material count and legal moves.
        
        Args:
            board: The chess board to analyze
            
        Returns:
            String representation of the position analysis
        """
        lines = []
        
        # Material count
        white_material = self._count_material(board, chess.WHITE)
        black_material = self._count_material(board, chess.BLACK)
        
        lines.append("Position Analysis:")
        lines.append(f"White material: {white_material}")
        lines.append(f"Black material: {black_material}")
        lines.append(f"Material difference: {white_material - black_material:+}")
        
        # Legal moves
        legal_moves = list(board.legal_moves)
        lines.append(f"Legal moves: {len(legal_moves)}")
        
        if legal_moves:
            # Show first few legal moves in UCI notation
            uci_moves = []
            for move in legal_moves[:10]:  # Show first 10 moves
                uci_moves.append(move.uci())
            
            lines.append(f"Sample moves: {', '.join(uci_moves)}")
            if len(legal_moves) > 10:
                lines.append(f"... and {len(legal_moves) - 10} more")
        
        lines.append("")
        
        # Add board
        lines.append(self.render_board(board, output_mode="string"))
        
        return "\n".join(lines)
    
    def _count_material(self, board: chess.Board, color: bool) -> int:
        """Count material value for a given color."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has no material value
        }
        
        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                total += piece_values[piece.piece_type]
        
        return total
