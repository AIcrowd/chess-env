"""
OpenAI chess agent implementation.

This agent uses OpenAI's API to make chess moves based on the current board state.
It follows the SPEC requirements for prompt templates and move parsing.
"""

import os
import time
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

import chess

from .base import ChessAgent

# Load environment variables from .env file
load_dotenv()


class OpenAIAgent(ChessAgent):
    """
    Chess agent that uses OpenAI's API to make moves.
    
    This agent follows the SPEC requirements:
    - Uses configurable prompt templates with placeholders
    - Parses moves in Standard Algebraic Notation (SAN)
    - Handles legal move validation
    - Configurable generation parameters
    """
    
    # Unicode chess piece characters (same as chess_renderer.py)
    UNICODE_PIECES = {
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
    
    # Default prompt template following SPEC requirements
    DEFAULT_PROMPT_TEMPLATE = """You are Magnus Carlsen, a chess grandmaster, with deep strategic understanding. Your task is to analyze the current chess position and select the best move available.

CURRENT BOARD STATE:
{board_utf}

POSITION INFORMATION:
- FEN notation: {FEN}
- Side to move: {side_to_move}
- Last move played: {last_move}

AVAILABLE MOVES:
- Legal moves in UCI notation: {legal_moves_uci}
- Legal moves in SAN notation: {legal_moves_san}

GAME HISTORY:
- Move history in UCI notation: {move_history_uci}
- Move history in SAN notation: {move_history_san}

INSTRUCTIONS:
1. Carefully analyze the position considering:
   - Material balance and piece activity
   - King safety and pawn structure
   - Control of key squares and files
   - Tactical opportunities and threats
   - Strategic long-term advantages

2. Select the best move from the available legal moves listed above.

3. IMPORTANT: You MUST respond with your chosen move in UCI notation (e.g., "e2e4", "g1f3", "e1g1") wrapped in <uci_move></uci_move> tags.

4. Do NOT use SAN notation (e.g., "e4", "Nf3", "O-O") in your response.

5. If you cannot find a good move or believe the position is lost, respond with <uci_move>resign</uci_move>

EXAMPLE RESPONSES:
- Correct: <uci_move>e2e4</uci_move>
- Correct: <uci_move>g1f3</uci_move>
- Correct: <uci_move>e1g1</uci_move> (kingside castling)
- Correct: <uci_move>resign</uci_move>

Remember: Always use UCI notation and wrap your response in <uci_move></uci_move> tags."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        fallback_behavior: str = "random_move",
        **kwargs
    ):
        """
        Initialize the OpenAI agent.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
            model: OpenAI model to use (e.g., "gpt-5-mini", "gpt-4", "gpt-3.5-turbo"). If None, will try to get from OPENAI_MODEL env var.
            prompt_template: Custom prompt template with placeholders. If None, uses default.
            temperature: Generation temperature (0.0 = deterministic, 1.0 = random)
            max_tokens: Maximum tokens to generate for the move
            timeout: API call timeout in seconds
            retry_attempts: Number of retry attempts for failed API calls
            retry_delay: Delay between retry attempts in seconds
            fallback_behavior: What to do when no valid move is found ("random_move" or "resign")
            **kwargs: Additional OpenAI API parameters
        """
        # Set up OpenAI client
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Load configuration from environment variables if not provided
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5")
        self.temperature = temperature or float(os.environ.get("OPENAI_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens or int(os.environ.get("OPENAI_MAX_TOKENS", "50"))
        self.timeout = timeout or float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Validate and set fallback behavior
        if fallback_behavior not in ["random_move", "resign"]:
            raise ValueError(
                f"Invalid fallback_behavior: {fallback_behavior}. "
                "Must be 'random_move' or 'resign'"
            )
        self.fallback_behavior = fallback_behavior
        
        # Load fallback behavior from environment variable if not provided
        if fallback_behavior is None:
            env_fallback = os.environ.get("OPENAI_FALLBACK_BEHAVIOR")
            if env_fallback:
                if env_fallback not in ["random_move", "resign"]:
                    raise ValueError(
                        f"Invalid OPENAI_FALLBACK_BEHAVIOR: {env_fallback}. "
                        "Must be 'random_move' or 'resign'"
                    )
                self.fallback_behavior = env_fallback
        
        # Additional OpenAI parameters - handle GPT-5 compatibility
        self.generation_params = {}
        
        # Add temperature only for models that support it
        if temperature is not None and not (self.model and "gpt-5" in self.model):
            self.generation_params["temperature"] = temperature
        
        # Add other parameters
        self.generation_params.update(kwargs)
        
        # Always use max_completion_tokens for the new Python API
        self.generation_params["max_completion_tokens"] = max_tokens
        
        # Prompt template
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        
        # Validate prompt template has required placeholders
        self._validate_prompt_template()
    
    def _validate_prompt_template(self):
        """Validate that the prompt template is valid and contains at least basic placeholders."""
        # Check for basic template validity
        if not self.prompt_template or not isinstance(self.prompt_template, str):
            raise ValueError("Prompt template must be a non-empty string")
        
        # Check for at least one placeholder to ensure it's a template
        if "{" not in self.prompt_template or "}" not in self.prompt_template:
            raise ValueError("Prompt template must contain at least one placeholder (e.g., {FEN})")
        
        # Check for balanced braces
        open_braces = self.prompt_template.count("{")
        close_braces = self.prompt_template.count("}")
        if open_braces != close_braces:
            raise ValueError("Prompt template has unbalanced braces - check for missing { or }")
        
        # Optional: Check for common issues with placeholder syntax
        import re
        placeholder_pattern = r'\{[^}]*\}'
        placeholders = re.findall(placeholder_pattern, self.prompt_template)
        
        # Warn about potentially problematic placeholders (but don't fail)
        for placeholder in placeholders:
            if placeholder in ["{}", "{ }", "{  }"]:
                print(f"Warning: Empty placeholder '{placeholder}' found in template")
        
        # Note: We don't require specific placeholders anymore - users can create custom templates
        # with only the variables they need
    

    
    def _format_prompt(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> str:
        """
        Format the prompt template with actual game data.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Formatted prompt string
        """
        # Get FEN representation
        fen = board.fen()
        
        # Get UTF board representation with Unicode chess pieces
        board_utf = self._render_board_unicode(board)
        
        # Get last move description
        if board.move_stack:
            last_move = board.move_stack[-1]
            # We need to get the SAN before the move was made
            # Create a temporary board to get the SAN
            temp_board = chess.Board()
            for move in board.move_stack[:-1]:
                temp_board.push(move)
            last_move_san = temp_board.san(last_move)
            last_side = "Black" if board.turn else "White"
            last_move_desc = f"{last_side} played {last_move_san}"
        else:
            last_move_desc = "(start of game)"
        
        # Format legal moves as both UCI and SAN lists
        legal_moves_uci = [move.uci() for move in legal_moves]
        legal_moves_san = [board.san(move) for move in legal_moves]
        legal_moves_uci_str = ", ".join(legal_moves_uci)
        legal_moves_san_str = ", ".join(legal_moves_san)
        
        # Format move history as both UCI and SAN
        if move_history:
            # Move history is already in UCI format
            move_history_uci_str = " ".join(move_history)
            
            # Convert UCI moves to SAN if possible
            try:
                history_board = chess.Board()
                move_history_san = []
                for uci_move in move_history:
                    try:
                        move = chess.Move.from_uci(uci_move)
                        san = history_board.san(move)
                        move_history_san.append(san)
                        history_board.push(move)
                    except:
                        move_history_san.append(uci_move)
                
                move_history_san_str = " ".join(move_history_san)
            except:
                move_history_san_str = " ".join(move_history)
        else:
            move_history_uci_str = "(no moves yet)"
            move_history_san_str = "(no moves yet)"
        
        # Format the prompt safely, handling missing placeholders
        try:
            prompt = self.prompt_template.format(
                board_utf=board_utf,
                FEN=fen,
                last_move=last_move_desc,
                legal_moves_uci=legal_moves_uci_str,
                legal_moves_san=legal_moves_san_str,
                move_history_uci=move_history_uci_str,
                move_history_san=move_history_san_str,
                side_to_move=side_to_move
            )
        except KeyError as e:
            # Handle missing placeholders gracefully
            missing_key = str(e).strip("'")
            print(f"Warning: Prompt template references placeholder '{missing_key}' that is not available")
            print("Available placeholders: board_utf, FEN, last_move, legal_moves_uci, legal_moves_san, move_history_uci, move_history_san, side_to_move")
            print("Consider updating your template or using the default template")
            
            # Fall back to a minimal template that should always work
            fallback_template = """You are playing chess. Choose the best move from the available legal moves.

Legal moves: {legal_moves_uci}

Respond with your move in UCI notation wrapped in <uci_move></uci_move> tags.

Example: <uci_move>e2e4</uci_move>"""
            
            prompt = fallback_template.format(legal_moves_uci=legal_moves_uci_str)
        
        return prompt
    
    def _render_board_unicode(self, board: chess.Board) -> str:
        """
        Render the chess board using Unicode chess pieces.
        
        Args:
            board: The chess board to render
            
        Returns:
            String representation of the board with Unicode pieces
        """
        lines = []
        
        # Board coordinates
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        # Add top coordinate line with proper alignment
        # Each square is 3 characters wide, so we need to center each letter
        coord_parts = []
        for file in files:
            coord_parts.append(f" {file} ")  # 3-character spacing to match board squares
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        # Calculate border width: 8 squares × 3 characters each = 24 characters
        border_width = len(files) * 3
        lines.append("   +" + "-" * border_width + "+")
        
        # Render board squares
        for rank_idx, rank in enumerate(ranks):
            line_parts = []
            
            # Add rank coordinate
            line_parts.append(f"{rank} |")
            
            # Add squares
            for file_idx, file in enumerate(files):
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                
                # Get piece symbol or empty square character
                if piece is None:
                    piece_char = "·"  # Empty square
                else:
                    piece_char = self.UNICODE_PIECES[piece.symbol()]
                
                # Format square
                square_str = f" {piece_char} "
                line_parts.append(square_str)
            
            # Add closing coordinate
            line_parts.append(f"| {rank}")
            lines.append("".join(line_parts))
        
        # Add bottom coordinate line
        lines.append("   +" + "-" * border_width + "+")
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        
        return "\n".join(lines)
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API to get the model's response.
        
        Args:
            prompt: Formatted prompt to send to the model
            
        Returns:
            Model's response text
            
        Raises:
            Exception: If API call fails after all retry attempts
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **self.generation_params,
                    timeout=self.timeout
                )
                
                # Extract the response content
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("OpenAI API returned None content")
                    return content.strip()
                else:
                    raise ValueError("Empty response from OpenAI API")
                    
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break
        
        # If we get here, all retry attempts failed
        raise Exception(f"OpenAI API call failed after {self.retry_attempts} attempts: {last_error}")
    
    def _parse_move(self, response: str, legal_moves: List[chess.Move], board: chess.Board) -> chess.Move:
        """
        Parse the model's response to extract a valid move.
        
        This method is super strict: the model must respond with a valid UCI move
        in the required <uci_move></uci_move> tags.
        
        Args:
            response: Raw response from the model
            legal_moves: List of legal moves available
            board: Current chess board state
            
        Returns:
            The chosen chess move
            
        Raises:
            ValueError: If no valid move can be parsed
        """
        # Clean the response
        response = response.strip()
        
        # Extract UCI move from tags - this is the ONLY acceptable format
        import re
        uci_pattern = r'<uci_move>(.*?)</uci_move>'
        uci_matches = re.findall(uci_pattern, response, re.IGNORECASE)
        
        # If no UCI tags found, raise error (not resignation)
        if not uci_matches:
            raise ValueError("Model did not respond with required UCI move tags")
        
        # Get the first UCI move from tags
        uci_move = uci_matches[0].strip()
        
        # Check for explicit resignation
        if uci_move.lower() == "resign":
            raise ValueError("Model chose to resign")
        
        # Try to parse the UCI move
        try:
            move = chess.Move.from_uci(uci_move)
        except Exception as e:
            # Invalid UCI format - raise error (not resignation)
            raise ValueError(f"Model provided invalid UCI format '{uci_move}'")
        
        # Check if the move is legal
        if move not in legal_moves:
            # Illegal move - raise error (not resignation)
            raise ValueError(f"Model provided illegal move '{uci_move}'")
        
        # Valid UCI move found
        return move
    
    def _extract_comment(self, response: str) -> str:
        """
        Extract the full comment from the model's response.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Full response as comment string
        """
        # Return the full response as the comment
        return response.strip()
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move | None, str | None]:
        """
        Choose the best move using OpenAI's API.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: The chosen chess move, or None to resign
            - optional_comment: Comment from the AI model explaining the move or resignation
            
        Raises:
            ValueError: If no legal moves are available
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Format the prompt
        prompt = self._format_prompt(board, legal_moves, move_history, side_to_move)
        
        # Call OpenAI API
        try:
            response = self._call_openai_api(prompt)
        except Exception as e:
            # If API call fails, fall back to first legal move
            print(f"Warning: OpenAI API call failed: {e}, using first legal move")
            return legal_moves[0], f"FALLBACK MOVE - OpenAI API failed: {e}"
        
        # Parse the response to get the move
        try:
            move = self._parse_move(response, legal_moves, board)
            # Use the full API response as the comment
            comment = self._extract_comment(response)
            return move, comment
        except ValueError as e:
            # Check if the model explicitly chose to resign
            if "resign" in str(e).lower():
                # Model explicitly chose to resign - always respect this choice
                return None, f"RESIGNATION - Model chose to resign. Full API response: {response}"
            
            # If parsing fails due to invalid/unparseable moves, handle according to fallback behavior
            if self.fallback_behavior == "resign":
                # Parsing failed and fallback behavior is resign
                return None, f"RESIGNATION - Unable to parse valid move: {e}. Full API response: {response}"
            else:
                # Parsing failed but fallback behavior is random_move - select a random legal move
                import random
                random_move = random.choice(legal_moves)
                print(f"Warning: Could not parse move from response: {e}, using random legal move")
                return random_move, f"RANDOM MOVE - Unable to parse move: {e}. Full API response: {response}"
    
    def update_prompt_template(self, new_template: str):
        """
        Update the prompt template.
        
        You can create custom templates with any combination of available placeholders:
        - {board_utf}: Visual board representation with Unicode pieces
        - {FEN}: FEN notation of the current position
        - {side_to_move}: Which side is to move ('White' or 'Black')
        - {legal_moves_uci}: Available moves in UCI notation
        - {legal_moves_san}: Available moves in SAN notation
        - {move_history_uci}: Game history in UCI notation
        - {move_history_san}: Game history in SAN notation
        - {last_move}: Description of the last move played
        
        Examples of custom templates:
        
        # Minimal template (only legal moves)
        "Choose the best move from: {legal_moves_uci}. Respond with <uci_move>move</uci_move>"
        
        # Position-focused template
        "Position: {FEN}\nYour turn: {side_to_move}\nLegal moves: {legal_moves_san}\nChoose: <uci_move>move</uci_move>"
        
        # Full-featured template (like the default)
        "Board:\n{board_utf}\nYour turn: {side_to_move}\nLegal moves: {legal_moves_uci}\nChoose: <uci_move>move</uci_move>"
        
        Args:
            new_template: New prompt template string with desired placeholders
        """
        self.prompt_template = new_template
        self._validate_prompt_template()
    
    def update_generation_params(self, **kwargs):
        """
        Update generation parameters.
        
        Args:
            **kwargs: New generation parameters
        """
        # Handle token parameter updates - always convert to max_completion_tokens
        if "max_tokens" in kwargs:
            # Convert max_tokens to max_completion_tokens for the new Python API
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        
        # Handle temperature updates for GPT-5 models
        if "temperature" in kwargs and (self.model and "gpt-5" in self.model):
            # Remove temperature for GPT-5 models as they don't support it
            kwargs.pop("temperature")
            print("   Note: Temperature parameter removed for GPT-5 model (not supported)")
        
        self.generation_params.update(kwargs)
        
        # Update instance variables for commonly used params
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_completion_tokens" in kwargs:
            self.max_tokens = kwargs["max_completion_tokens"]
    
    def update_fallback_behavior(self, behavior: str):
        """
        Update the fallback behavior.
        
        Args:
            behavior: New fallback behavior ("random_move" or "resign")
            
        Raises:
            ValueError: If behavior is invalid
        """
        if behavior not in ["random_move", "resign"]:
            raise ValueError(
                f"Invalid fallback_behavior: {behavior}. "
                "Must be 'random_move' or 'resign'"
            )
        self.fallback_behavior = behavior
    
    def get_prompt_template(self) -> str:
        """Get the current prompt template."""
        return self.prompt_template
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get the current generation parameters."""
        return self.generation_params.copy()
    
    def get_fallback_behavior(self) -> str:
        """Get the current fallback behavior."""
        return self.fallback_behavior
    
    def test_connection(self) -> bool:
        """
        Test the OpenAI API connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use minimal parameters for connection test
            test_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "timeout": 10
            }
            
            # Always use max_completion_tokens for the new Python API
            test_params["max_completion_tokens"] = 5
            
            response = self.client.chat.completions.create(**test_params)
            return True
        except Exception as e:
            print(f"OpenAI API connection test failed: {e}")
            return False
