"""
Stockfish chess agent implementation.

This agent uses the Stockfish chess engine to make strong chess moves.
"""

import os
import platform
import subprocess
from typing import Any, Dict, List, Optional

import chess
from dotenv import load_dotenv

from .base import ChessAgent

# Load environment variables from .env file
load_dotenv()


class StockfishAgent(ChessAgent):
    """
    Chess agent that uses the Stockfish chess engine.
    
    This agent requires the Stockfish binary to be installed on the system.
    It automatically detects common installation paths or can be configured
    with a custom path via environment variable STOCKFISH_PATH.
    
    The agent provides robust cleanup and can be used as a context manager:
    
    ```python
    # Automatic cleanup with context manager
    with StockfishAgent() as agent:
        move = agent.choose_move(board, legal_moves, [], "White")
    
    # Manual cleanup
    agent = StockfishAgent()
    try:
        move = agent.choose_move(board, legal_moves, [], "White")
    finally:
        agent.close()
    ```
    """
    
    # Common Stockfish binary paths for different operating systems
    COMMON_PATHS = {
        "darwin": [  # macOS
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "/usr/bin/stockfish",
        ],
        "linux": [  # Linux
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/usr/games/stockfish",
        ],
        "win32": [  # Windows
            "C:\\Program Files\\Stockfish\\stockfish.exe",
            "C:\\Program Files (x86)\\Stockfish\\stockfish.exe",
            "stockfish.exe",  # If in PATH
        ],
    }
    
    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 15,
        skill_level: int = 20,
        elo_rating: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        time_limit_ms: Optional[int] = None,
        hash_size_mb: int = 128,
        threads: int = 1,
    ):
        """
        Initialize the Stockfish agent.
        
        Args:
            stockfish_path: Path to Stockfish binary. If None, will auto-detect.
            depth: Search depth for Stockfish (default: 15)
            skill_level: Skill level 0-20 (default: 20, highest)
            elo_rating: ELO rating to limit strength (ignores skill_level if set)
            parameters: Additional Stockfish parameters as dict
            time_limit_ms: Time limit per move in milliseconds
            hash_size_mb: Hash table size in MB (default: 128)
            threads: Number of threads to use (default: 1)
            
        Raises:
            RuntimeError: If Stockfish binary cannot be found or started
        """
        self.depth = depth
        self.skill_level = skill_level
        self.elo_rating = elo_rating
        self.time_limit_ms = time_limit_ms
        self.hash_size_mb = hash_size_mb
        self.threads = threads
        
        # Find Stockfish binary
        self.stockfish_path = self._find_stockfish_binary(stockfish_path)
        
        # Initialize Stockfish process
        self._stockfish = None
        self._initialize_stockfish()
        
        # Set initial parameters
        self._set_parameters(parameters or {})
    
    def _find_stockfish_binary(self, custom_path: Optional[str]) -> str:
        """
        Find the Stockfish binary path.
        
        Args:
            custom_path: Custom path provided by user
            
        Returns:
            Path to Stockfish binary
            
        Raises:
            RuntimeError: If Stockfish binary cannot be found
        """
        # Check custom path first
        if custom_path:
            if os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
                return custom_path
            else:
                raise RuntimeError(f"Custom Stockfish path is not executable: {custom_path}")
        
        # Check environment variable
        env_path = os.environ.get("STOCKFISH_PATH")
        if env_path:
            if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
                return env_path
            else:
                raise RuntimeError(f"STOCKFISH_PATH environment variable points to non-executable file: {env_path}")
        
        # Auto-detect based on platform
        system = platform.system().lower()
        if system == "darwin":
            paths = self.COMMON_PATHS["darwin"]
        elif system == "linux":
            paths = self.COMMON_PATHS["linux"]
        elif system == "windows":
            paths = self.COMMON_PATHS["win32"]
        else:
            paths = []
        
        # Check common paths
        for path in paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["which", "stockfish"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return path
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # If we get here, Stockfish was not found
        error_msg = (
            "Stockfish binary not found. Please install Stockfish or set the STOCKFISH_PATH "
            "environment variable.\n\n"
            "Installation instructions:\n"
            "  macOS: brew install stockfish\n"
            "  Ubuntu/Debian: sudo apt install stockfish\n"
            "  Windows: Download from https://stockfishchess.org/download/\n"
            "  Or set STOCKFISH_PATH environment variable to point to your Stockfish binary."
        )
        raise RuntimeError(error_msg)
    
    def _initialize_stockfish(self):
        """Initialize the Stockfish process."""
        try:
            self._stockfish = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Test if Stockfish is working
            self._send_command("uci")
            self._send_command("isready")
            
            # Wait for readyok with timeout
            response = self._read_response(timeout=5.0)  # 5 second timeout for initialization
            if "readyok" not in response:
                raise RuntimeError("Stockfish did not respond with 'readyok'")
            
            # Start a new game context to clear internal engine state
            self._send_command("ucinewgame")
                
        except Exception as e:
            if hasattr(self, '_stockfish') and self._stockfish:
                try:
                    self._stockfish.terminate()
                except Exception:
                    pass  # Ignore errors during cleanup
                self._stockfish = None
            raise RuntimeError(f"Failed to initialize Stockfish: {e}")
    
    def _send_command(self, command: str):
        """Send a command to Stockfish."""
        if not hasattr(self, '_stockfish') or not self._stockfish:
            raise RuntimeError("Stockfish process not initialized")
        
        self._stockfish.stdin.write(command + "\n")
        self._stockfish.stdin.flush()
    
    def _read_response(self, timeout: float = 1.0) -> str:
        """Read response from Stockfish with timeout."""
        if not hasattr(self, '_stockfish') or not self._stockfish:
            raise RuntimeError("Stockfish process not initialized")
        
        response = ""
        try:
            # Simple timeout-based reading
            import time
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Try to read a line
                if self._stockfish.stdout.readable():
                    line = self._stockfish.stdout.readline()
                    if line:
                        response += line
                        if "readyok" in line or "bestmove" in line:
                            break
                    else:
                        # No more data available
                        break
                else:
                    # Small delay to avoid busy waiting
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Warning: Error reading from Stockfish: {e}")
        
        return response
    
    def _set_parameters(self, additional_params: Dict[str, Any]):
        """Set Stockfish engine parameters."""
        # Set basic parameters
        params = {
            "Hash": self.hash_size_mb,
            "Threads": self.threads,
            "Skill Level": self.skill_level,
        }
        
        # Add additional parameters
        params.update(additional_params)
        
        # Apply parameters
        for param, value in params.items():
            self._send_command(f"setoption name {param} value {value}")
        
        # Set ELO rating if specified
        if self.elo_rating is not None:
            self._send_command(f"setoption name UCI_LimitStrength value true")
            self._send_command(f"setoption name UCI_Elo value {self.elo_rating}")
    
    def _set_position(self, board: chess.Board):
        """Set the current position in Stockfish using exact FEN and wait for readiness."""
        # If it's a fresh game (no moves yet), inform engine explicitly
        if not board.move_stack:
            self._send_command("ucinewgame")
        # Use exact FEN to avoid any discrepancy in castling/en passant rights
        fen = board.fen()
        self._send_command(f"position fen {fen}")
        # Ensure engine processed position before searching
        self._send_command("isready")
        self._read_response(timeout=1.0)
    
    def _get_best_move(self) -> str:
        """Get the best move from Stockfish."""
        if self.time_limit_ms:
            self._send_command(f"go movetime {self.time_limit_ms}")
        else:
            self._send_command(f"go depth {self.depth}")
        
        # Use a reasonable timeout for move calculation
        if self.time_limit_ms:
            timeout = max(0.1, (self.time_limit_ms / 1000.0) * 2.0)  # 2x the time limit, minimum 0.1 second
        else:
            timeout = max(0.1, (self.depth / 20.0))  # Reasonable timeout based on depth
        response = self._read_response(timeout=timeout)
        
        # Parse best move from response
        for line in response.split('\n'):
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        
        raise RuntimeError("Stockfish did not return a best move")
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move | None, str | None]:
        """
        Choose the best move using Stockfish engine.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: The best move according to Stockfish, or None to resign
            - optional_comment: Comment describing the move evaluation or resignation
            
        Raises:
            RuntimeError: If Stockfish fails to provide a move
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        try:
            # Set the current position
            self._set_position(board)
            
            # Get best move from Stockfish
            best_move_uci = self._get_best_move()
            
            # Convert UCI string to chess.Move
            best_move = chess.Move.from_uci(best_move_uci)
            
            # Verify the move is legal
            if best_move not in legal_moves:
                # If Stockfish suggests an illegal move, fall back to first legal move
                print(f"Warning: Stockfish suggested illegal move {best_move_uci}, using first legal move")
                return legal_moves[0], "FALLBACK MOVE - Stockfish suggested illegal move"
            
            # Create a comment about the move
            comment = f"Stockfish engine move (depth: {self.depth}, skill: {self.skill_level})"
            if self.elo_rating:
                comment += f", ELO limited to {self.elo_rating}"
            
            return best_move, comment
            
        except Exception as e:
            # Fallback to first legal move if Stockfish fails
            print(f"Warning: Stockfish failed: {e}, using first legal move")
            return legal_moves[0], f"FALLBACK MOVE - Stockfish failed: {e}"
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """
        Update Stockfish engine parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self._set_parameters(parameters)
    
    def set_skill_level(self, skill_level: int):
        """
        Set Stockfish skill level (0-20).
        
        Args:
            skill_level: Skill level from 0 (weakest) to 20 (strongest)
        """
        if not 0 <= skill_level <= 20:
            raise ValueError("Skill level must be between 0 and 20")
        
        self.skill_level = skill_level
        self._send_command(f"setoption name Skill Level value {skill_level}")
    
    def set_elo_rating(self, elo_rating: int):
        """
        Set Stockfish ELO rating limit.
        
        Args:
            elo_rating: ELO rating to limit strength
        """
        self.elo_rating = elo_rating
        self._send_command("setoption name UCI_LimitStrength value true")
        self._send_command(f"setoption name UCI_Elo value {elo_rating}")
    
    def set_depth(self, depth: int):
        """
        Set search depth for Stockfish.
        
        Args:
            depth: Search depth
        """
        self.depth = depth
    
    def set_time_limit(self, time_limit_ms: int):
        """
        Set time limit per move.
        
        Args:
            time_limit_ms: Time limit in milliseconds
        """
        self.time_limit_ms = time_limit_ms
    
    def is_initialized(self) -> bool:
        """
        Check if the Stockfish agent is properly initialized.
        
        Returns:
            True if Stockfish process is running, False otherwise
        """
        return hasattr(self, '_stockfish') and self._stockfish is not None and self._stockfish.poll() is None
    
    def __del__(self):
        """Clean up Stockfish process on deletion."""
        # Use hasattr to safely check if the attribute exists
        if hasattr(self, '_stockfish') and self._stockfish:
            try:
                self._stockfish.terminate()
                self._stockfish.wait(timeout=1)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    self._stockfish.kill()
                except Exception:
                    pass  # Ignore errors during cleanup
    
    def close(self):
        """Explicitly close the Stockfish process."""
        if hasattr(self, '_stockfish') and self._stockfish:
            try:
                self._stockfish.terminate()
                self._stockfish.wait(timeout=1)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    self._stockfish.kill()
                except Exception:
                    pass  # Ignore errors during cleanup
            finally:
                self._stockfish = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
