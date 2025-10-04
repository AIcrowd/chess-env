"""
Hugging Face chess agent implementation.

This agent uses Hugging Face Inference API models to make chess moves based on
the current board state. It mirrors the `OpenAIAgent` structure: prompt
templating, strict UCI move parsing via <uci_move></uci_move> tags, and
graceful fallbacks.
"""

import os
import time
from typing import Any, Dict, List, Optional

import chess
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .base import ChessAgent

# Load environment variables from .env file
load_dotenv()


class HuggingFaceAgent(ChessAgent):
    """
    Chess agent that uses Hugging Face Inference API chat models.

    Key behaviors:
    - Configurable prompt template using placeholders
    - Strict parsing of <uci_move></uci_move>
    - Legal move validation and configurable fallback
    """

    UNICODE_PIECES = {
        'P': '♙',
        'R': '♖',
        'N': '♘',
        'B': '♗',
        'Q': '♕',
        'K': '♔',
        'p': '♟',
        'r': '♜',
        'n': '♞',
        'b': '♝',
        'q': '♛',
        'k': '♚',
    }

    DEFAULT_PROMPT_TEMPLATE = """You are Magnus Carlsen, a chess grandmaster. Analyze the position and return the best move.

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
1) Carefully consider tactics and strategy. 2) Choose the best move from legal moves.
IMPORTANT: Respond ONLY with the chosen move in UCI notation wrapped in <uci_move></uci_move> tags.
Examples: <uci_move>e2e4</uci_move>, <uci_move>g1f3</uci_move>, <uci_move>e1g1</uci_move>, or <uci_move>resign</uci_move>.
"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_token: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        fallback_behavior: str = "random_move",
        **kwargs,
    ):
        """
        Initialize the Hugging Face agent.

        Args:
            model: Hugging Face model id (e.g., "deepseek-ai/DeepSeek-V3-0324").
            api_token: HF token. If None, uses HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.
            prompt_template: Custom prompt template.
            max_tokens: Maximum new tokens to generate.
            timeout: API call timeout in seconds.
            retry_attempts: Number of retries on failure.
            retry_delay: Delay between retries.
            fallback_behavior: "random_move" or "resign".
            **kwargs: Additional parameters passed to client.chat.completions.create.
        """
        self.api_token = (
            api_token
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        if not self.api_token:
            raise ValueError(
                "Hugging Face API token not provided. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN."
            )

        self.client = InferenceClient(token=self.api_token)

        self.model = model or os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-V3-0324")
        self.max_tokens = max_tokens or int(os.environ.get("HF_MAX_TOKENS", "64"))
        self.timeout = timeout or float(os.environ.get("HF_TIMEOUT", "30.0"))
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        if fallback_behavior not in ["random_move", "resign"]:
            raise ValueError("fallback_behavior must be 'random_move' or 'resign'")
        self.fallback_behavior = fallback_behavior

        self.generation_params: Dict[str, Any] = {}
        if self.max_tokens is not None:
            self.generation_params["max_tokens"] = self.max_tokens
        # Allow passthrough kwargs for model-specific params (e.g., temperature if supported)
        self.generation_params.update(kwargs)

        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self._validate_prompt_template()

    def _validate_prompt_template(self):
        if not self.prompt_template or not isinstance(self.prompt_template, str):
            raise ValueError("Prompt template must be a non-empty string")
        open_braces = self.prompt_template.count("{")
        close_braces = self.prompt_template.count("}")
        if open_braces != close_braces:
            raise ValueError("Prompt template has unbalanced braces")

    def _render_board_unicode(self, board: chess.Board) -> str:
        lines = []
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']

        coord_parts = []
        for file in files:
            coord_parts.append(f" {file} ")
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        border_width = len(files) * 3
        lines.append("   +" + "-" * border_width + "+")

        for rank in ranks:
            line_parts = []
            line_parts.append(f"{rank} |")
            for file in files:
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                piece_char = "·" if piece is None else self.UNICODE_PIECES[piece.symbol()]
                line_parts.append(f" {piece_char} ")
            line_parts.append(f"| {rank}")
            lines.append("".join(line_parts))

        lines.append("   +" + "-" * border_width + "+")
        lines.append(coord_line)
        return "\n".join(lines)

    def _format_prompt(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> str:
        fen = board.fen()
        board_utf = self._render_board_unicode(board)

        if board.move_stack:
            last_move = board.move_stack[-1]
            temp_board = chess.Board()
            for mv in board.move_stack[:-1]:
                temp_board.push(mv)
            last_move_san = temp_board.san(last_move)
            last_side = "Black" if board.turn else "White"
            last_move_desc = f"{last_side} played {last_move_san}"
        else:
            last_move_desc = "(start of game)"

        legal_moves_uci = [m.uci() for m in legal_moves]
        legal_moves_san = [board.san(m) for m in legal_moves]
        legal_moves_uci_str = ", ".join(legal_moves_uci)
        legal_moves_san_str = ", ".join(legal_moves_san)

        if move_history:
            move_history_uci_str = " ".join(move_history)
            try:
                history_board = chess.Board()
                history_san: List[str] = []
                for u in move_history:
                    try:
                        mv = chess.Move.from_uci(u)
                        san = history_board.san(mv)
                        history_san.append(san)
                        history_board.push(mv)
                    except Exception:
                        history_san.append(u)
                move_history_san_str = " ".join(history_san)
            except Exception:
                move_history_san_str = " ".join(move_history)
        else:
            move_history_uci_str = "(no moves yet)"
            move_history_san_str = "(no moves yet)"

        try:
            prompt = self.prompt_template.format(
                board_utf=board_utf,
                FEN=fen,
                last_move=last_move_desc,
                legal_moves_uci=legal_moves_uci_str,
                legal_moves_san=legal_moves_san_str,
                move_history_uci=move_history_uci_str,
                move_history_san=move_history_san_str,
                side_to_move=side_to_move,
            )
        except KeyError:
            fallback = (
                "You are playing chess. Choose the best move from: {moves}.\n"
                "Respond with <uci_move>...</uci_move>"
            )
            prompt = fallback.format(moves=legal_moves_uci_str)
        return prompt

    def _call_hf_api(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **self.generation_params,
                )
                if completion and completion.choices and completion.choices[0].message:
                    content = completion.choices[0].message.content
                    if content is None:
                        raise ValueError("Empty content from HF API")
                    return content.strip()
                raise ValueError("Empty response from HF API")
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                break
        raise Exception(f"HF API call failed after {self.retry_attempts} attempts: {last_error}")

    def _parse_move(self, response: str, legal_moves: List[chess.Move]) -> chess.Move:
        response = response.strip()
        import re
        matches = re.findall(r"<uci_move>(.*?)</uci_move>", response, re.IGNORECASE)
        if not matches:
            raise ValueError("Model did not respond with required UCI move tags")
        uci = matches[0].strip()
        if uci.lower() == "resign":
            raise ValueError("Model chose to resign")
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            raise ValueError(f"Model provided invalid UCI format '{uci}'")
        if move not in legal_moves:
            raise ValueError(f"Model provided illegal move '{uci}'")
        return move

    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move | None, str | None]:
        if not legal_moves:
            raise ValueError("No legal moves available")

        prompt = self._format_prompt(board, legal_moves, move_history, side_to_move)
        try:
            response = self._call_hf_api(prompt)
        except Exception as e:
            # API failure -> fallback first legal move
            return legal_moves[0], f"FALLBACK MOVE - HF API failed: {e}"

        try:
            move = self._parse_move(response, legal_moves)
            return move, response.strip()
        except ValueError as e:
            if "resign" in str(e).lower():
                return None, f"RESIGNATION - Model chose to resign. Full API response: {response}"
            if self.fallback_behavior == "resign":
                return None, f"RESIGNATION - Unable to parse valid move: {e}. Full API response: {response}"
            import random
            random_move = random.choice(legal_moves)
            return random_move, f"RANDOM MOVE - Unable to parse move: {e}. Full API response: {response}"

    def update_prompt_template(self, new_template: str):
        self.prompt_template = new_template
        self._validate_prompt_template()

    def update_generation_params(self, **kwargs):
        self.generation_params.update(kwargs)
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]

    def get_prompt_template(self) -> str:
        return self.prompt_template

    def get_generation_params(self) -> Dict[str, Any]:
        return self.generation_params.copy()

    def get_fallback_behavior(self) -> str:
        return self.fallback_behavior


