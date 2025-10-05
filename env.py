import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine

from agents import ChessAgent, RandomAgent
from chess_renderer import RICH_AVAILABLE, ChessRenderer


class ChessEnvironment:
    """Chess environment for running games between two agents."""

    def __init__(
        self,
        agent1: ChessAgent,
        agent2: ChessAgent,
        max_moves: int = 200,
        time_limit: float = 10.0,
        initial_fen: Optional[str] = None,
    ):
        """
        Initialize the chess environment.

        Args:
            agent1: First agent (plays as White)
            agent2: Second agent (plays as Black)
            max_moves: Maximum number of moves before declaring a draw
            time_limit: Time limit per move in seconds
            initial_fen: Optional FEN string to start the game from. If None, uses standard starting position.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_moves = max_moves
        self.time_limit = time_limit
        self.board = chess.Board()
        self.move_history = []
        self.move_comments = []  # Store comments for each move
        self.game_result = None
        # Track illegal move attempts and total move attempts (excluding resignations)
        self.white_illegal_attempts = 0
        self.black_illegal_attempts = 0
        self.white_move_attempts = 0
        self.black_move_attempts = 0
        
        # Set initial position if provided
        if initial_fen is not None:
            self._initial_fen = initial_fen
            self.reset(initial_fen)
        else:
            self._initial_fen = chess.STARTING_FEN
        
        # Initialize renderer
        self.renderer = ChessRenderer()

    def reset(self, fen: str = chess.STARTING_FEN):
        """Reset the board to a new position."""
        self.board = chess.Board(fen)
        self.move_history = []
        self.move_comments = []
        self.game_result = None
        self._initial_fen = fen
        # Reset attempt counters
        self.white_illegal_attempts = 0
        self.black_illegal_attempts = 0
        self.white_move_attempts = 0
        self.black_move_attempts = 0

    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves for the current position."""
        return list(self.board.legal_moves)

    def get_legal_moves_uci(self) -> List[str]:
        """Get all legal moves in Universal Chess Interface (UCI) notation."""
        return [move.uci() for move in self.board.legal_moves]

    def get_fen(self) -> str:
        """Get the current board position in FEN notation."""
        return self.board.fen()

    def get_side_to_move(self) -> str:
        """Get whose turn it is ('White' or 'Black')."""
        return "White" if self.board.turn else "Black"

    def get_last_move(self) -> Optional[str]:
        """Get the last move played in UCI notation."""
        if not self.move_history:
            return None
        return self.move_history[-1]
    
    def get_last_move_comment(self) -> Optional[str]:
        """Get the comment for the last move played."""
        if not self.move_comments:
            return None
        return self.move_comments[-1]
    
    def get_move_comment(self, move_index: int) -> Optional[str]:
        """Get the comment for a specific move by index."""
        if 0 <= move_index < len(self.move_comments):
            return self.move_comments[move_index]
        return None

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()

    def get_game_result(self) -> Optional[str]:
        """Get the game result if the game is over."""
        if not self.is_game_over():
            return None

        # Use the correct python-chess API
        if hasattr(self.board, 'outcome'):
            # Newer versions of python-chess
            outcome = self.board.outcome()
            if outcome is None:
                return "Draw"
            elif outcome.winner is None:
                return "Draw"
            else:
                return "White wins" if outcome.winner else "Black wins"
        else:
            # Older versions of python-chess
            if self.board.is_checkmate():
                return "Black wins" if self.board.turn else "White wins"
            elif self.board.is_stalemate():
                return "Draw"
            elif self.board.is_insufficient_material():
                return "Draw"
            elif self.board.is_fifty_moves():
                return "Draw"
            elif self.board.is_repetition():
                return "Draw"
            else:
                return "Draw"
    
    def get_game_termination_reason(self) -> str:
        """Get the specific reason why the game ended."""
        if not self.is_game_over():
            return "Game not over"
        
        if self.board.is_checkmate():
            return "Checkmate"
        elif self.board.is_stalemate():
            return "Stalemate"
        elif self.board.is_insufficient_material():
            return "Insufficient material"
        elif self.board.is_fifty_moves():
            return "Fifty-move rule"
        elif self.board.is_repetition():
            return "Threefold repetition"
        elif self.board.is_variant_draw():
            return "Variant draw"
        else:
            return "Unknown"

    def play_move(self, move: chess.Move, comment: str | None = None) -> bool:
        """
        Play a move on the board.

        Args:
            move: The move to play
            comment: Optional comment about the move

        Returns:
            True if the move was successful, False otherwise
        """
        if move in self.board.legal_moves:
            uci_move = move.uci()
            self.board.push(move)
            self.move_history.append(uci_move)
            self.move_comments.append(comment)
            return True
        return False

    def play_agent_move(self, agent: ChessAgent, side: str) -> Optional[chess.Move]:
        """
        Get a move from an agent and play it.

        Args:
            agent: The agent to get a move from
            side: Which side the agent is playing ('White' or 'Black')

        Returns:
            The move that was played, or None if failed
        """
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            return None

        try:
            # Get move from agent
            start_time = time.time()
            move_result = agent.choose_move(self.board, legal_moves, self.move_history, side)
            move_time = time.time() - start_time
            
            # Handle the new tuple return format
            if isinstance(move_result, tuple):
                move, comment = move_result
            else:
                # Backward compatibility for old agents
                move = move_result
                comment = None

            # Check for resignation
            if move is None:
                print(f"❌ {side} chose to resign")
                return None

            # Check time limit
            if move_time > self.time_limit:
                print(
                    f"Warning: {side} took {move_time:.2f}s (limit: {self.time_limit}s)"
                )

            # Validate and play the move
            if move in legal_moves:
                # Count this as an attempted move for the side
                if side == "White":
                    self.white_move_attempts += 1
                else:
                    self.black_move_attempts += 1
                self.play_move(move, comment)
                return move
            else:
                print(f"Warning: {side} returned illegal move {move}")
                # Count this as an attempted move and an illegal attempt for the side
                if side == "White":
                    self.white_move_attempts += 1
                    self.white_illegal_attempts += 1
                else:
                    self.black_move_attempts += 1
                    self.black_illegal_attempts += 1
                return None

        except Exception as e:
            print(f"Error getting move from {side}: {e}")
            return None

    def play_game(self, verbose: bool = True, progress_callback: callable = None) -> Dict[str, Any]:
        """
        Play a complete game between the two agents.

        Args:
            verbose: Whether to print game progress
            progress_callback: Optional callback function called after each move with (move_count, current_side)

        Returns:
            Dictionary containing game results and statistics
        """
        self.reset()
        move_count = 0

        if verbose:
            print(f"Starting new game: {self.agent1.__class__.__name__} (White) vs {self.agent2.__class__.__name__} (Black)")
            print(f"Initial position: {self.get_fen()}")
            print()
            # Show initial board
            print("Initial Position:")
            self.renderer.render_board(self.board, output_mode="clean")
            print()

        while not self.is_game_over() and move_count < self.max_moves:
            current_side = self.get_side_to_move()
            current_agent = self.agent1 if current_side == "White" else self.agent2

            if verbose:
                print(f"Move {move_count + 1}: {current_side}'s turn")
                
                # Show legal moves (only first 5 for clarity)
                legal_moves = self.get_legal_moves_uci()
                display_moves = legal_moves[:5]
                if len(legal_moves) > 5:
                    display_moves.append(f"... and {len(legal_moves) - 5} more")
                print(f"Legal moves: {', '.join(display_moves)}")

            # Get and play move
            move = self.play_agent_move(current_agent, current_side)
            if move is None:
                # Agent failed to provide a valid move or resigned
                if current_side == "White":
                    result = "Black wins (White resigned)"
                else:
                    result = "White wins (Black resigned)"
                if verbose:
                    print(f"❌ {current_side} failed to provide a valid move. {result}")
                break

            if verbose:
                move_uci = self.get_last_move()
                move_comment = self.get_last_move_comment()
                
                print(f"✅ {current_side} plays: {move_uci}")
                if move_comment:
                    print(f"   Comment: {move_comment}")
                
                # Show board after move (only rich version for clarity)
                print(f"\nPosition after {move_count + 1}. {move_uci}:")
                self.renderer.render_board(self.board, last_move=move, output_mode="clean")
                print()
                
                # Show move summary
                print(f"Move {move_count + 1}: {move_uci} | Side: {current_side} | Agent: {current_agent.__class__.__name__}")
                if move_comment:
                    print(f"   Analysis: {move_comment}")
                print("-" * 60)

            move_count += 1
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(move_count, current_side)

        # Game ended
        if 'result' not in locals():
            result = self.get_game_result()
            if result is None:
                result = "Draw (max moves reached)"

        if verbose:
            print(f"\n🎯 Game Over: {result}")
            print(f"📊 Total moves: {move_count}")
            
            # Show termination reason if game ended normally
            if self.is_game_over():
                termination_reason = self.get_game_termination_reason()
                print(f"🏁 Termination: {termination_reason}")
            
            # Show final board position
            print(f"\n📋 Final Position:")
            self.renderer.render_board(self.board, output_mode="clean")
            print()
            
            print(f"📝 Move history: {' '.join(self.move_history)}")

        # Compile results
        game_stats = {
            "result": result,
            "moves_played": move_count,
            "move_history": self.move_history.copy(),
            "move_comments": self.move_comments.copy(),
            "final_fen": self.get_fen(),
            "white_agent": self.agent1.__class__.__name__,
            "black_agent": self.agent2.__class__.__name__,
            "game_over_reason": (
                self.get_game_termination_reason() if self.is_game_over() else "max_moves"
            ),
            # Illegal move attempt statistics
            "white_illegal_attempts": self.white_illegal_attempts,
            "black_illegal_attempts": self.black_illegal_attempts,
            "white_move_attempts": self.white_move_attempts,
            "black_move_attempts": self.black_move_attempts,
        }

        return game_stats

    def get_pgn(self) -> str:
        """Get the game in PGN format."""
        if not self.move_history:
            return ""
        
        # Convert UCI moves to SAN for PGN export (PGN standard uses SAN)
        san_moves = []
        temp_board = chess.Board(self._initial_fen)
        
        for i, uci_move in enumerate(self.move_history):
            move = chess.Move.from_uci(uci_move)
            try:
                san_move = temp_board.san(move)
                # Add comment if available
                comment = self.move_comments[i] if i < len(self.move_comments) and self.move_comments[i] else None
                if comment:
                    san_moves.append(f"{san_move} {{{comment}}}")
                else:
                    san_moves.append(san_move)
                temp_board.push(move)
            except:
                # Fallback to UCI if SAN conversion fails
                comment = self.move_comments[i] if i < len(self.move_comments) and self.move_comments[i] else None
                if comment:
                    san_moves.append(f"{uci_move} {{{comment}}}")
                else:
                    san_moves.append(uci_move)
                temp_board.push(move)
        
        pgn_lines = [
            '[Event "Chess Game"]',
            f'[White "{self.agent1.__class__.__name__}"]',
            f'[Black "{self.agent2.__class__.__name__}"]',
            f'[Result "{self._get_pgn_result()}"]',
            "",
            " ".join(san_moves),
        ]
        
        # Add final result if game is over
        if self.is_game_over():
            result = self._get_pgn_result()
            if result != "*":
                pgn_lines.append(f" {result}")
        
        return "\n".join(pgn_lines)

    def export_pgn_file(self, filename: str, include_metadata: bool = True) -> bool:
        """
        Export the game to a PGN file.
        
        Args:
            filename: Name of the file to save (with or without .pgn extension)
            include_metadata: Whether to include additional metadata in the PGN
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure filename has .pgn extension
            if not filename.endswith('.pgn'):
                filename += '.pgn'
            
            # Generate PGN content
            pgn_content = self._generate_pgn_content(include_metadata)
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(pgn_content)
            
            return True
            
        except Exception as e:
            print(f"Error exporting PGN file: {e}")
            return False
    
    def _generate_pgn_content(self, include_metadata: bool = True) -> str:
        """
        Generate PGN content with optional metadata.
        
        Args:
            include_metadata: Whether to include additional metadata
            
        Returns:
            Formatted PGN string
        """
        if not self.move_history:
            return ""
        
        # Convert UCI moves to SAN for PGN export
        san_moves = []
        temp_board = chess.Board(self._initial_fen)
        
        for i, uci_move in enumerate(self.move_history):
            move = chess.Move.from_uci(uci_move)
            try:
                san_move = temp_board.san(move)
                # Add comment if available
                comment = self.move_comments[i] if i < len(self.move_comments) and self.move_comments[i] else None
                if comment:
                    san_moves.append(f"{san_move} {{{comment}}}")
                else:
                    san_moves.append(san_move)
                temp_board.push(move)
            except:
                # Fallback to UCI if SAN conversion fails
                comment = self.move_comments[i] if i < len(self.move_comments) and self.move_comments[i] else None
                if comment:
                    san_moves.append(f"{uci_move} {{{comment}}}")
                else:
                    san_moves.append(uci_move)
                temp_board.push(move)
        
        # Derive display names including model identifiers when available
        def _agent_display_name(agent_obj: Any) -> str:
            try:
                # Prefer explicit model attribute if present (HF/OpenAI agents)
                model_id = getattr(agent_obj, "model", None)
                if model_id:
                    return f"{agent_obj.__class__.__name__} ({model_id})"
            except Exception:
                pass
            return agent_obj.__class__.__name__

        white_name = _agent_display_name(self.agent1)
        black_name = _agent_display_name(self.agent2)

        # Basic PGN headers
        pgn_lines = [
            '[Event "Chess Game"]',
            f'[Site "Chess Environment"]',
            f'[Date "{self._get_current_date()}"]',
            f'[Round "1"]',
            f'[White "{white_name}"]',
            f'[Black "{black_name}"]',
            f'[Result "{self._get_pgn_result()}"]',
        ]
        
        # Add optional metadata
        if include_metadata:
            pgn_lines.extend([
                f'[WhiteType "program"]',
                f'[BlackType "program"]',
                f'[TimeControl "-"]',
                f'[Termination "{self._get_termination_reason()}"]',
                f'[Moves "{len(self.move_history)}"]',
                f'[InitialFEN "{self._get_initial_fen()}"]',
                f'[FinalFEN "{self.get_fen()}"]',
            ])
        
        # Add moves
        pgn_lines.append("")
        pgn_lines.append(" ".join(san_moves))
        
        # Add final result if game is over
        if self.is_game_over():
            result = self._get_pgn_result()
            if result != "*":
                pgn_lines.append(f" {result}")
        
        return "\n".join(pgn_lines)
    
    def _get_current_date(self) -> str:
        """Get current date in PGN format (YYYY.MM.DD)."""
        import datetime
        now = datetime.datetime.now()
        return now.strftime("%Y.%m.%d")
    
    def _get_pgn_result(self) -> str:
        """Get PGN result string."""
        if not self.is_game_over():
            return "*"
        
        result = self.get_game_result()
        if result == "White wins":
            return "1-0"
        elif result == "Black wins":
            return "0-1"
        elif result == "Draw":
            return "1/2-1/2"
        else:
            return "*"
    
    def _get_termination_reason(self) -> str:
        """Get termination reason for PGN."""
        if not self.is_game_over():
            return "unterminated"
        
        termination_reason = self.get_game_termination_reason()
        if termination_reason == "Checkmate":
            return "checkmate"
        elif termination_reason == "Stalemate":
            return "stalemate"
        elif termination_reason == "Insufficient material":
            return "insufficient material"
        elif termination_reason == "Fifty-move rule":
            return "fifty-move rule"
        elif termination_reason == "Threefold repetition":
            return "repetition"
        elif termination_reason == "Variant draw":
            return "variant draw"
        else:
            return "normal"
    
    def _get_initial_fen(self) -> str:
        """Get the initial FEN position."""
        # If we have a custom starting position, return it
        # Otherwise return the standard starting position
        if hasattr(self, '_initial_fen') and self._initial_fen != chess.STARTING_FEN:
            return self._initial_fen
        return chess.STARTING_FEN

    def display_board(self, highlight_last_move: bool = True, clean: bool = True) -> str:
        """
        Display the current chess board using Unicode characters.
        
        Args:
            highlight_last_move: Whether to highlight the last move played
            clean: Whether to use clean rendering (no duplicate displays)
            
        Returns:
            String representation of the chess board
        """
        last_move = None
        if highlight_last_move and self.move_history:
            # Get the last move from the board's move stack
            if len(self.board.move_stack) > 0:
                last_move = self.board.move_stack[-1]
        
        if clean:
            return self.renderer.render_board(self.board, last_move, output_mode="clean")
        else:
            return self.renderer.render_board(self.board, last_move, output_mode="auto")
    
    def display_game_state(self, show_move_history: bool = True) -> str:
        """
        Display the complete game state including board and information.
        
        Args:
            show_move_history: Whether to show the move history
            
        Returns:
            String representation of the complete game state
        """
        return self.renderer.render_game_state(
            self.board,
            move_history=self.move_history if show_move_history else None,
            side_to_move=self.get_side_to_move(),
            game_result=self.get_game_result()
        )
    
    def display_position_analysis(self) -> str:
        """
        Display position analysis including material count and legal moves.
        
        Returns:
            String representation of the position analysis
        """
        return self.renderer.render_position_analysis(self.board)
    
    def display_move_sequence(self, moves: List[chess.Move], 
                            start_fen: str = None) -> str:
        """
        Display a sequence of moves showing the board after each move.
        
        Args:
            moves: List of moves to show
            start_fen: Optional starting FEN position
            
        Returns:
            String representation of the move sequence
        """
        return self.renderer.render_move_sequence(self.board, moves, start_fen)
    
    def set_renderer_options(self, show_coordinates: bool = None, 
                           show_move_numbers: bool = None,
                           empty_square_char: str = None,
                           use_rich: bool = None) -> None:
        """
        Configure renderer display options.
        
        Args:
            show_coordinates: Whether to show file/rank coordinates
            show_move_numbers: Whether to show move numbers in the display
            empty_square_char: Character to show for empty squares
            use_rich: Whether to use rich CLI for enhanced rendering
        """
        if show_coordinates is not None:
            self.renderer.show_coordinates = show_coordinates
        if show_move_numbers is not None:
            self.renderer.show_move_numbers = show_move_numbers
        if empty_square_char is not None:
            self.renderer.empty_square_char = empty_square_char
        if use_rich is not None:
            self.renderer.use_rich = use_rich and RICH_AVAILABLE


def main():
    """Example usage of the chess environment."""
    # Create two random agents
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    # Create environment
    env = ChessEnvironment(agent1, agent2, max_moves=100, time_limit=5.0)

    # Play a game
    print("Playing a game between two random agents...")
    result = env.play_game(verbose=True)

    print(f"\nGame Summary:")
    print(f"Result: {result['result']}")
    print(f"Moves played: {result['moves_played']}")
    print(f"White agent: {result['white_agent']}")
    print(f"Black agent: {result['black_agent']}")

    # Get PGN
    pgn = env.get_pgn()
    print(f"\nPGN:\n{pgn}")


if __name__ == "__main__":
    main()
