#!/usr/bin/env python3
"""
Enhanced gameplay script with CLI configuration and parallel game execution

This script demonstrates:
- CLI configuration for agent selection, game parameters, and execution options
- Support for multiple games with parallel execution using p_tqdm
- Aggregated results and statistics
- Single PGN file output for all games

Usage examples:
  python run_game.py --agent1 openai-gpt-4o --agent2 stockfish-skill1-depth2
  python run_game.py --agent1 stockfish-skill5-depth10 --agent2 openai-gpt-4o --num-games 10 --verbose
  python run_game.py --agent1 openai-gpt-5-mini --agent2 openai-gpt-4o-mini --num-games 5
  python run_game.py --help
"""

import datetime
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess
import chess.engine
import click
import trueskill
from p_tqdm import p_map
from rich.console import Console
from rich.json import JSON
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from agents import HuggingFaceAgent, OpenAIAgent, StockfishAgent
from env import ChessEnvironment


@dataclass
class GameResult:
    """Container for individual game results."""
    game_id: int
    result: str
    moves_played: int
    white_agent: str
    black_agent: str
    game_over_reason: str
    final_fen: str
    pgn_content: str
    # Per-game engine-based metrics (averaged per side)
    white_accuracy_pct: float = 0.0
    black_accuracy_pct: float = 0.0
    white_acpl: float = 0.0
    black_acpl: float = 0.0
    pgn_path: str = ""
    # Illegal move attempt metrics
    white_illegal_attempts: int = 0
    black_illegal_attempts: int = 0
    white_move_attempts: int = 0
    black_move_attempts: int = 0


class _StockfishAnalyzer:
    """Lightweight Stockfish-based analyzer to compute engine-match accuracy and ACPL.

    Accuracy is defined as percentage of moves that match Stockfish's top choice at
    the configured search settings. ACPL (Average Centipawn Loss) is computed from
    the pre-move and post-move evaluations from White's perspective.
    """

    def __init__(self, stockfish_path: str | None = None, depth: int = 20, movetime_ms: int | None = 1000):
        path = stockfish_path or os.getenv("STOCKFISH_PATH") or "stockfish"
        # Start UCI engine via python-chess
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth
        self.movetime_ms = movetime_ms

    def _get_limit(self) -> chess.engine.Limit:
        """Create engine limit based on configured time or depth."""
        return chess.engine.Limit(
                time=self.movetime_ms / 1000.0,
                depth=self.depth if self.depth else None
        )

    def _analyse_position(self, board: chess.Board) -> tuple[int, str | None]:
        """Analyze position and return (eval_cp, best_move_uci).
        
        Returns:
            tuple[int, str | None]: (centipawn evaluation, best move in UCI format)
        """
        limit = self._get_limit()
        info = self.engine.analyse(board, limit)
        
        # Extract evaluation
        score = info["score"].relative
        if score is None:
            import warnings
            warnings.warn("Score is None, returning 1000")
            eval_cp = 1000
        else:
            # Cap mate and cp scores at ±1000 (similar to lichess/chess.com)
            cp = score.score(mate_score=1000)
            if cp is not None:
                eval_cp = int(max(-1000, min(1000, cp)))
            else:
                import warnings
                warnings.warn("CP score is None after conversion, returning 1000")
                eval_cp = 1000
        
        # Extract best move from principal variation
        best_move = None
        if "pv" in info and len(info["pv"]) > 0:
            best_move = info["pv"][0].uci()
        
        return eval_cp, best_move

    def analyze_game(self, moves_uci: list[str], initial_fen: str | None = None) -> dict:
        board = chess.Board(initial_fen) if initial_fen else chess.Board()
        white_moves = 0
        black_moves = 0
        white_matches = 0
        black_matches = 0
        white_cpl_sum = 0
        black_cpl_sum = 0

        try:
            for uci in moves_uci:
                # Analyze position before the move (gets both eval and best move)
                eval_before, best_move = self._analyse_position(board)

                move = chess.Move.from_uci(uci)
                is_white = board.turn  # True if White to move

                # Check if move matches engine's best move
                if best_move and best_move == uci:
                    if is_white:
                        white_matches += 1
                    else:
                        black_matches += 1

                # Apply actual move
                board.push(move)
                
                # Evaluate after the move (from opponent's perspective now)
                eval_after, _ = self._analyse_position(board)
                
                # Convert eval_after to the mover's perspective by negating
                # (since the turn has flipped)
                eval_after_mover_pov = -eval_after
                
                # CPL is the loss in centipawns from the mover's perspective
                cpl = max(0, eval_before - eval_after_mover_pov)

                if is_white:
                    white_moves += 1
                    white_cpl_sum += cpl
                else:
                    black_moves += 1
                    black_cpl_sum += cpl

            white_accuracy = (white_matches / white_moves * 100.0) if white_moves else 0.0
            black_accuracy = (black_matches / black_moves * 100.0) if black_moves else 0.0
            white_acpl = (white_cpl_sum / white_moves) if white_moves else 1000  # 1000 = MAX_ACPL 
            black_acpl = (black_cpl_sum / black_moves) if black_moves else 1000  # 1000 = MAX_ACPL

            return {
                "white_accuracy_pct": white_accuracy,
                "black_accuracy_pct": black_accuracy,
                "white_acpl": white_acpl,
                "black_acpl": black_acpl,
            }
        finally:
            try:
                self.engine.quit()
            except Exception:
                pass


class TournamentProgressTracker:
    """Tracks progress across multiple parallel chess games."""
    
    def __init__(self, total_games: int, max_moves: int):
        self.total_games = total_games
        self.max_moves = max_moves
        self.lock = threading.Lock()
        self.game_progress = {}  # game_id -> moves_played
        self.completed_games = 0
        self.total_moves_played = 0
        self.start_time = time.time()
        
    def update_game_progress(self, game_id: int, moves_played: int):
        """Update progress for a specific game."""
        with self.lock:
            if game_id not in self.game_progress:
                self.total_moves_played += moves_played
            else:
                self.total_moves_played += (moves_played - self.game_progress[game_id])
            self.game_progress[game_id] = moves_played
            
            # Print a simple progress update every 10 moves or so
            if moves_played % 10 == 0 or moves_played == 1:
                print(f"📊 Game {game_id}: {moves_played} moves | Total: {self.total_moves_played} moves across all games")
    
    def mark_game_completed(self, game_id: int):
        """Mark a game as completed."""
        with self.lock:
            # Always mark as completed if we have any record of this game
            if game_id in self.game_progress or game_id <= self.total_games:
                self.completed_games += 1
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            active_games = len(self.game_progress)
            avg_moves_per_game = self.total_moves_played / max(active_games, 1)
            
            return {
                "completed_games": self.completed_games,
                "active_games": active_games,
                "total_moves_played": self.total_moves_played,
                "avg_moves_per_game": avg_moves_per_game,
                "elapsed_time": elapsed_time,
                "estimated_total_moves": self.total_games * self.max_moves
            }
    
    def render_progress_display(self) -> Panel:
        """Render a rich progress display."""
        stats = self.get_progress_stats()
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Progress overview
        progress_pct = (stats["completed_games"] / self.total_games) * 100
        table.add_row("🎯 Progress", f"{stats['completed_games']}/{self.total_games} games ({progress_pct:.1f}%)")
        
        # Move statistics
        moves_pct = (stats["total_moves_played"] / stats["estimated_total_moves"]) * 100 if stats["estimated_total_moves"] > 0 else 0
        table.add_row("📊 Moves", f"{stats['total_moves_played']} total ({moves_pct:.1f}% of max)")
        table.add_row("📈 Avg Moves/Game", f"{stats['avg_moves_per_game']:.1f}")
        
        # Time statistics
        elapsed_min = stats["elapsed_time"] / 60
        table.add_row("⏱️  Elapsed Time", f"{elapsed_min:.1f} minutes")
        
        # Active games
        table.add_row("🔄 Active Games", f"{stats['active_games']}")
        
        return Panel(table, title="🏆 Tournament Progress", border_style="green")


class AgentFactory:
    """Factory class for creating chess agents from string specifications."""
    
    # Cache for Stockfish agents to avoid recreating processes
    _stockfish_cache = {}
    
    @staticmethod
    def create_agent(agent_spec: str) -> Any:
        """
        Create an agent from a string specification.
        
        Args:
            agent_spec: String like "openai-gpt-4o" or "stockfish-skill1-depth2"
            
        Returns:
            Chess agent instance
            
        Raises:
            ValueError: If agent specification is invalid
        """
        if agent_spec.startswith("openai-"):
            # Hardcoded OpenAI agent configurations
            if agent_spec == "openai-gpt-4o":
                return OpenAIAgent(
                    model="gpt-4o",
                    max_tokens=500
                )
            elif agent_spec == "openai-gpt-4o-mini":
                return OpenAIAgent(
                    model="gpt-4o-mini",
                    max_tokens=500
                )
            elif agent_spec == "openai-gpt-5-mini":
                return OpenAIAgent(
                    model="gpt-5-mini",
                    max_tokens=500
                )
            elif agent_spec == "openai-gpt-5":
                return OpenAIAgent(
                    model="gpt-5",
                    max_tokens=500
                )
            else:
                raise ValueError(f"Unsupported OpenAI agent: {agent_spec}. Supported: openai-gpt-4o, openai-gpt-4o-mini, openai-gpt-5-mini, openai-gpt-5")
            
        elif agent_spec.startswith("stockfish-"):
            # Parse Stockfish agent specification
            parts = agent_spec.split("-")
            skill_level = 0  # Use skill 0 for faster responses
            depth = 1        # Use depth 1 for faster responses
            time_limit_ms = 1000
            
            for part in parts[1:]:
                if part.startswith("skill"):
                    # Support both forms: skill10 and skill=10
                    m = re.search(r"skill(?:=)?(\d+)", part)
                    if m:
                        try:
                            skill_level = int(m.group(1))
                        except ValueError:
                            pass
                elif part.startswith("depth"):
                    # Support both forms: depth2 and depth=2
                    m = re.search(r"depth(?:=)?(\d+)", part)
                    if m:
                        try:
                            depth = int(m.group(1))
                        except ValueError:
                            pass
                elif part.startswith("time"):
                    # Support both forms: time1000 and time=1000 (milliseconds)
                    m = re.search(r"time(?:=)?(\d+)", part)
                    if m:
                        try:
                            time_limit_ms = int(m.group(1))
                        except ValueError:
                            pass
            
            # Create a cache key for this configuration
            cache_key = f"stockfish-skill{skill_level}-depth{depth}-time{time_limit_ms}"
            
            # Check if we have a cached instance
            if cache_key in AgentFactory._stockfish_cache:
                # Return a copy of the cached agent (to avoid sharing state between games)
                cached_agent = AgentFactory._stockfish_cache[cache_key]
                # Create a new instance with the same parameters but fresh state
                return StockfishAgent(
                    skill_level=cached_agent.skill_level,
                    depth=cached_agent.depth,
                    time_limit_ms=cached_agent.time_limit_ms,
                    stockfish_path=cached_agent.stockfish_path
                )
            else:
                # Create new agent and cache it
                agent = StockfishAgent(
                    skill_level=skill_level,
                    depth=depth,
                    time_limit_ms=time_limit_ms,
                    hash_size_mb=1,  # Minimal hash for faster startup
                    threads=1         # Single thread for faster responses
                )
                AgentFactory._stockfish_cache[cache_key] = agent
                return agent
            
        elif agent_spec.startswith("hf-"):
            # Hugging Face agent configurations (hf-<model_alias> or hf-<model_id>)
            # Examples: hf-deepseek, hf-deepseek-v3, hf-meta-llama-3.1-8b-instruct
            alias = agent_spec[len("hf-"):]
            # Simple alias mapping; users can supply full repo ids too
            alias_map = {
                # DeepSeek (not <10B but kept for completeness/examples)
                "deepseek": "deepseek-ai/DeepSeek-V3-0324",
                "deepseek-v3": "deepseek-ai/DeepSeek-V3-0324",

                # Focus: <10B class aliases
                # Llama 3 / 3.1 8B Instruct
                "llama-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
                "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
                "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
                "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",

                # Qwen 7B Instruct
                "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",

                # Mistral 7B Instruct
                "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",

                # Phi 3.x Mini
                "phi-3-mini": "microsoft/Phi-3-mini-128k-instruct",
                "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",

                # Gemma 7B IT
                "gemma-7b": "google/gemma-7b-it",
            }
            model_id = alias_map.get(alias, alias)
            return HuggingFaceAgent(model=model_id, max_tokens=500)
        else:
            raise ValueError(f"Unknown agent type: {agent_spec}")


def play_single_game(
    game_id: int,
    agent1_spec: str,
    agent2_spec: str,
    max_moves: int,
    time_limit: float,
    verbose: bool,
    progress_tracker: TournamentProgressTracker = None,
    force_white_spec: str | None = None,
    force_black_spec: str | None = None,
) -> GameResult:
    """
    Play a single chess game between two agents.
    
    Args:
        game_id: Unique identifier for the game
        agent1_spec: String specification for first agent
        agent2_spec: String specification for second agent
        max_moves: Maximum number of moves allowed
        time_limit: Time limit per move in seconds
        verbose: Whether to print detailed game progress
        
    Returns:
        GameResult object containing game outcome and metadata
    """
    try:
        # Create agents
        if force_white_spec is not None and force_black_spec is not None:
            white_agent = AgentFactory.create_agent(force_white_spec)
            black_agent = AgentFactory.create_agent(force_black_spec)
            white_name = force_white_spec
            black_name = force_black_spec
        else:
            agent1 = AgentFactory.create_agent(agent1_spec)
            agent2 = AgentFactory.create_agent(agent2_spec)
            # Randomly assign White and Black
            agents = [agent1, agent2]
            random.shuffle(agents)
            white_agent, black_agent = agents
            # Get agent names for display
            white_name = f"{agent1_spec}" if white_agent == agent1 else f"{agent2_spec}"
            black_name = f"{agent2_spec}" if white_agent == agent1 else f"{agent1_spec}"
        
        if verbose:
            print(f"🎲 Game {game_id}: {white_name} (White) vs {black_name} (Black)")
        
        # Create environment
        env = ChessEnvironment(
            agent1=white_agent,
            agent2=black_agent,
            max_moves=max_moves,
            time_limit=time_limit
        )
        
        # Update Stockfish agents with the time limit if they are Stockfish agents
        if hasattr(white_agent, 'set_time_limit'):
            # For Stockfish agents, use a much more aggressive time limit
            # since they should respond in milliseconds, not seconds
            if "stockfish" in str(type(white_agent)).lower():
                white_agent.set_time_limit(50)  # 50ms for Stockfish
            else:
                white_agent.set_time_limit(int(time_limit * 1000))
        if hasattr(black_agent, 'set_time_limit'):
            # For Stockfish agents, use a much more aggressive time limit
            if "stockfish" in str(type(black_agent)).lower():
                black_agent.set_time_limit(50)  # 50ms for Stockfish
            else:
                black_agent.set_time_limit(int(time_limit * 1000))
        
        # Play the game with progress tracking
        if progress_tracker and not verbose:
            # Define progress callback for this game
            def game_progress_callback(move_count, current_side):
                progress_tracker.update_game_progress(game_id, move_count)
            
            # Use standard play_game with progress callback
            result = env.play_game(verbose=verbose, progress_callback=game_progress_callback)
        else:
            # Use standard play_game for verbose mode
            result = env.play_game(verbose=verbose)
        
        # Generate PGN content
        pgn_content = env._generate_pgn_content(include_metadata=True)
        
        # Compute engine-based accuracy metrics
        try:
            analyzer = _StockfishAnalyzer()
            analysis = analyzer.analyze_game(result.get('move_history', []))
        except Exception:
            analysis = {
                "white_accuracy_pct": 0.0,
                "black_accuracy_pct": 0.0,
                "white_acpl": 0.0,
                "black_acpl": 0.0,
            }

        # Create game result
        game_result = GameResult(
            game_id=game_id,
            result=result['result'],
            moves_played=result['moves_played'],
            white_agent=white_name,
            black_agent=black_name,
            game_over_reason=result['game_over_reason'],
            final_fen=result['final_fen'],
            pgn_content=pgn_content,
            white_accuracy_pct=analysis["white_accuracy_pct"],
            black_accuracy_pct=analysis["black_accuracy_pct"],
            white_acpl=analysis["white_acpl"],
            black_acpl=analysis["black_acpl"],
            white_illegal_attempts=result.get('white_illegal_attempts', 0),
            black_illegal_attempts=result.get('black_illegal_attempts', 0),
            white_move_attempts=result.get('white_move_attempts', result['moves_played']//2 + (1 if result['moves_played'] % 2 == 1 else 0)),
            black_move_attempts=result.get('black_move_attempts', result['moves_played']//2),
        )
        
        # Mark game as completed in progress tracker
        if progress_tracker:
            progress_tracker.mark_game_completed(game_id)
            # Also update final move count
            progress_tracker.update_game_progress(game_id, result['moves_played'])
        
        if verbose:
            print(f"✅ Game {game_id} completed: {result['result']} in {result['moves_played']} moves")
        
        return game_result
        
    except Exception as e:
        if verbose:
            print(f"❌ Game {game_id} failed: {str(e)}")
        
        # Return error result
        return GameResult(
            game_id=game_id,
            result="ERROR",
            moves_played=0,
            white_agent=agent1_spec,
            black_agent=agent2_spec,
            game_over_reason=f"Error: {str(e)}",
            final_fen="",
            pgn_content=""
        )





def aggregate_results(game_results: List[GameResult]) -> Dict[str, Any]:
    """
    Aggregate results from multiple games.
    
    Args:
        game_results: List of GameResult objects
        
    Returns:
        Dictionary containing aggregated statistics
    """
    if not game_results:
        return {}
    
    # Count wins by agent
    agent_wins = {}
    agent_games = {}
    agent_white_games = {}
    agent_black_games = {}
    total_moves = 0
    successful_games = 0
    # Accuracy aggregates
    agent_accuracy_sum = {}
    agent_accuracy_count = {}
    agent_acpl_sum = {}
    agent_acpl_count = {}
    # Illegal attempts aggregates
    agent_illegal_attempts = {}
    agent_move_attempts = {}
    
    for result in game_results:
        if result.result == "ERROR":
            continue
            
        successful_games += 1
        total_moves += result.moves_played
        
        # Count games played by each agent
        for agent in [result.white_agent, result.black_agent]:
            if agent not in agent_games:
                agent_games[agent] = 0
                agent_wins[agent] = 0
                agent_white_games[agent] = 0
                agent_black_games[agent] = 0
                agent_accuracy_sum[agent] = 0.0
                agent_accuracy_count[agent] = 0
                agent_acpl_sum[agent] = 0.0
                agent_acpl_count[agent] = 0
            agent_games[agent] += 1
            
            # Count white/black assignments
            if agent == result.white_agent:
                agent_white_games[agent] += 1
            else:
                agent_black_games[agent] += 1

        # Accumulate accuracy and ACPL per agent (per game averages)
        agent_accuracy_sum[result.white_agent] += result.white_accuracy_pct
        agent_accuracy_count[result.white_agent] += 1
        agent_acpl_sum[result.white_agent] += result.white_acpl
        agent_acpl_count[result.white_agent] += 1

        agent_accuracy_sum[result.black_agent] += result.black_accuracy_pct
        agent_accuracy_count[result.black_agent] += 1
        agent_acpl_sum[result.black_agent] += result.black_acpl
        agent_acpl_count[result.black_agent] += 1

        # Initialize illegal tracking dicts if needed
        for agent in [result.white_agent, result.black_agent]:
            if agent not in agent_illegal_attempts:
                agent_illegal_attempts[agent] = 0
                agent_move_attempts[agent] = 0

        # Accumulate illegal attempts and move attempts per agent (per game totals)
        agent_illegal_attempts[result.white_agent] += getattr(result, 'white_illegal_attempts', 0)
        agent_illegal_attempts[result.black_agent] += getattr(result, 'black_illegal_attempts', 0)
        agent_move_attempts[result.white_agent] += getattr(result, 'white_move_attempts', 0)
        agent_move_attempts[result.black_agent] += getattr(result, 'black_move_attempts', 0)
        
        # Count wins, draws, and other outcomes
        if "wins" in result.result:
            winner = result.white_agent if "White wins" in result.result else result.black_agent
            agent_wins[winner] += 1
        elif "Draw" in result.result:
            # For draws, both agents get 0.5 points
            agent_wins[result.white_agent] += 0.5
            agent_wins[result.black_agent] += 0.5
    
    # Calculate statistics
    stats = {
        "total_games": len(game_results),
        "successful_games": successful_games,
        "failed_games": len(game_results) - successful_games,
        "total_moves": total_moves,
        "average_moves_per_game": total_moves / successful_games if successful_games > 0 else 0,
        "agent_wins": agent_wins,
        "agent_games": agent_games,
        "agent_white_games": agent_white_games,
        "agent_black_games": agent_black_games,
        "win_rates": {},
        "agent_avg_accuracy_pct": {},
        "agent_avg_acpl": {},
        "agent_illegal_attempts": agent_illegal_attempts,
        "agent_move_attempts": agent_move_attempts,
        "agent_illegal_pct": {},
    }
    
    # Calculate win rates
    for agent, wins in agent_wins.items():
        games_played = agent_games.get(agent, 0)
        stats["win_rates"][agent] = (wins / games_played * 100) if games_played > 0 else 0
        # Accuracy/ACPL averages
        if agent_accuracy_count.get(agent, 0) > 0:
            stats["agent_avg_accuracy_pct"][agent] = agent_accuracy_sum[agent] / agent_accuracy_count[agent]
        else:
            stats["agent_avg_accuracy_pct"][agent] = 0.0
        if agent_acpl_count.get(agent, 0) > 0:
            stats["agent_avg_acpl"][agent] = agent_acpl_sum[agent] / agent_acpl_count[agent]
        else:
            stats["agent_avg_acpl"][agent] = 0.0
        # Illegal percentage
        attempts = agent_move_attempts.get(agent, 0)
        illegal = agent_illegal_attempts.get(agent, 0)
        stats["agent_illegal_pct"][agent] = (illegal / attempts * 100.0) if attempts > 0 else 0.0
    
    return stats


def save_combined_pgn(game_results: List[GameResult], output_file: str = "games.pgn"):
    """
    Save all games to a single PGN file.
    
    Args:
        game_results: List of GameResult objects
        output_file: Output PGN filename
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(game_results):
            if result.result == "ERROR":
                continue
                
            # Add game separator
            if i > 0:
                f.write("\n\n")
            
            # Write game PGN content
            f.write(result.pgn_content)
    
    print(f"💾 Combined PGN file saved to: {output_file}")


def print_summary_stats(stats: Dict[str, Any], game_results: List[GameResult]):
    """
    Print summary statistics for all games.
    
    Args:
        stats: Aggregated statistics dictionary
        game_results: List of GameResult objects for detailed analysis
    """
    console = Console()
    
    print("\n" + "=" * 80)
    print("📊 GAME SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"🎯 Total Games: {stats['total_games']}")
    print(f"✅ Successful Games: {stats['successful_games']}")
    print(f"❌ Failed Games: {stats['failed_games']}")
    print(f"📈 Total Moves: {stats['total_moves']}")
    print(f"📊 Average Moves per Game: {stats['average_moves_per_game']:.1f}")
    
    print(f"\n🏆 AGENT PERFORMANCE:")
    print("-" * 40)
    
    for agent, wins in stats['agent_wins'].items():
        games_played = stats['agent_games'].get(agent, 0)
        white_games = stats['agent_white_games'].get(agent, 0)
        black_games = stats['agent_black_games'].get(agent, 0)
        win_rate = stats['win_rates'].get(agent, 0)
        illegal_pct = stats.get('agent_illegal_pct', {}).get(agent, 0.0)
        illegal_attempts = stats.get('agent_illegal_attempts', {}).get(agent, 0)
        total_attempts = stats.get('agent_move_attempts', {}).get(agent, 0)
        print(f"🎮 {agent}:")
        print(f"   Games Played: {games_played}")
        print(f"   As White: {white_games}")
        print(f"   As Black: {black_games}")
        print(f"   Points: {wins}")
        print(f"   Win Rate: {win_rate:.1f}%")
        # Accuracy details
        avg_acc = stats.get('agent_avg_accuracy_pct', {}).get(agent, 0.0)
        avg_acpl = stats.get('agent_avg_acpl', {}).get(agent, 0.0)
        print(f"   Accuracy (engine match): {avg_acc:.1f}%")
        print(f"   ACPL: {avg_acpl:.1f}")
        print(f"   Illegal Moves: {illegal_attempts}/{total_attempts} ({illegal_pct:.2f}%)")
        print()
    
    # Print prettified JSON stats
    print("\n" + "=" * 80)
    print("📋 DETAILED STATISTICS (JSON)")
    print("=" * 80)
    
    # Create a clean stats dict for JSON output
    json_stats = {
        "tournament_summary": {
            "total_games": stats['total_games'],
            "successful_games": stats['successful_games'],
            "failed_games": stats['failed_games'],
            "total_moves": stats['total_moves'],
            "average_moves_per_game": round(stats['average_moves_per_game'], 2)
        },
        "agent_performance": {},
        "game_outcomes": {
            "wins": sum(1 for result in game_results if "wins" in result.result),
            "draws": sum(1 for result in game_results if "Draw" in result.result),
            "max_moves_reached": sum(1 for result in game_results if "max moves" in result.result),
            "other": sum(1 for result in game_results if result.result not in ["ERROR"] and "wins" not in result.result and "Draw" not in result.result)
        }
    }
    
    # Add agent performance data
    for agent, wins in stats['agent_wins'].items():
        games_played = stats['agent_games'].get(agent, 0)
        white_games = stats['agent_white_games'].get(agent, 0)
        black_games = stats['agent_black_games'].get(agent, 0)
        win_rate = stats['win_rates'].get(agent, 0)
        json_stats["agent_performance"][agent] = {
            "games_played": games_played,
            "as_white": white_games,
            "as_black": black_games,
            "points": wins,  # Can be fractional for draws
            "win_rate": round(win_rate, 2),
            "win_percentage": f"{win_rate:.1f}%",
            "avg_accuracy_pct": round(stats.get('agent_avg_accuracy_pct', {}).get(agent, 0.0), 2),
            "avg_acpl": round(stats.get('agent_avg_acpl', {}).get(agent, 0.0), 2),
            "illegal_attempts": stats.get('agent_illegal_attempts', {}).get(agent, 0),
            "move_attempts": stats.get('agent_move_attempts', {}).get(agent, 0),
            "illegal_pct": round(stats.get('agent_illegal_pct', {}).get(agent, 0.0), 2),
        }
    
    # Print prettified JSON using rich
    console.print(JSON(json.dumps(json_stats, indent=2)))


# =====================
# N-agent Tournament
# =====================

@dataclass
class AgentState:
    spec: str
    rating: trueskill.Rating
    games: int = 0
    as_white: int = 0
    as_black: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    accuracy_sum: float = 0.0
    acpl_sum: float = 0.0
    illegal_attempts_sum: int = 0
    move_attempts_sum: int = 0
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    @property
    def conservative(self) -> float:
        return self.rating.mu - 3 * self.rating.sigma


def _result_to_code(result_str: str) -> str:
    if "White wins" in result_str:
        return "1-0"
    if "Black wins" in result_str:
        return "0-1"
    if "Draw" in result_str:
        return "1-1"
    return "*"


def _choose_colors_soft_balance(a_state: AgentState, b_state: AgentState) -> Tuple[str, str]:
    a_delta = a_state.as_white - a_state.as_black
    b_delta = b_state.as_white - b_state.as_black
    if a_delta > b_delta:
        # Give White to B to reduce imbalance
        return b_state.spec, a_state.spec
    else:
        return a_state.spec, b_state.spec


def run_tournament(
    agent_specs: List[str],
    num_games: int,
    max_moves: int,
    time_limit: float,
    scheduler: str,
    parallelism: int,
    output_dir: str,
    max_games_per_agent: int,
    verbose: bool,
):
    # Prepare output directories
    out_dir = Path(output_dir)
    pgn_dir = out_dir / "pgns"
    pgn_dir.mkdir(parents=True, exist_ok=True)

    # TrueSkill environment
    ts_env = trueskill.TrueSkill(draw_probability=0.1)
    trueskill.setup()

    # Agent states
    agent_states: Dict[str, AgentState] = {
        spec: AgentState(spec=spec, rating=ts_env.Rating()) for spec in agent_specs
    }

    # Head-to-head matrix
    h2h: Dict[Tuple[str, str], Dict[str, int]] = {}

    games: List[Dict[str, Any]] = []
    results_raw: List[GameResult] = []
    total_scheduled = 0
    caps_relaxed = False

    # Progress tracker
    tracker = None if verbose else TournamentProgressTracker(num_games, max_moves)

    def can_schedule_pair(a: str, b: str) -> bool:
        if a == b:
            return False
        if not caps_relaxed and max_games_per_agent > 0:
            if agent_states[a].games >= max_games_per_agent:
                return False
            if agent_states[b].games >= max_games_per_agent:
                return False
        return True

    while total_scheduled < num_games:
        # Determine batch size
        remaining = num_games - total_scheduled
        batch_target = min(parallelism, remaining)

        # Select pairs
        candidate_pairs: List[Tuple[str, str, float]] = []
        for a, b in combinations(agent_specs, 2):
            if not can_schedule_pair(a, b):
                continue
            qa = trueskill.quality_1vs1(agent_states[a].rating, agent_states[b].rating)
            candidate_pairs.append((a, b, qa))
        # Sort by quality descending
        candidate_pairs.sort(key=lambda x: x[2], reverse=True)

        # Build a batch allowing agent reuse to meet the parallelism target.
        # We still respect max_games_per_agent unless caps are relaxed.
        batch = []
        planned_counts: Dict[str, int] = {}
        # Keep cycling candidate pairs until we either fill the batch or can no longer add pairs
        while len(batch) < batch_target and candidate_pairs:
            added_any = False
            for a, b, q in candidate_pairs:
                if len(batch) >= batch_target:
                    break
                # Respect caps if not relaxed (consider already planned games in this batch)
                if not caps_relaxed and max_games_per_agent > 0:
                    if agent_states[a].games + planned_counts.get(a, 0) >= max_games_per_agent:
                        continue
                    if agent_states[b].games + planned_counts.get(b, 0) >= max_games_per_agent:
                        continue
                a_state = agent_states[a]
                b_state = agent_states[b]
                white_spec, black_spec = _choose_colors_soft_balance(a_state, b_state)
                batch.append((a, b, white_spec, black_spec))
                planned_counts[a] = planned_counts.get(a, 0) + 1
                planned_counts[b] = planned_counts.get(b, 0) + 1
                added_any = True
            if not added_any:
                # If we couldn't add anything in this pass, try relaxing caps once
                if not caps_relaxed and max_games_per_agent > 0:
                    caps_relaxed = True
                    continue
                else:
                    break

        # If we cannot fill any game, relax caps once and retry; otherwise stop
        if len(batch) == 0:
            if not caps_relaxed and max_games_per_agent > 0:
                caps_relaxed = True
                continue
            else:
                # Impossible to schedule more games
                break

        # Run batch
        args_list = []
        for i, (a, b, w, bl) in enumerate(batch):
            gid = total_scheduled + i + 1
            args_list.append((gid, a, b, max_moves, time_limit, False if tracker else verbose, tracker, w, bl))

        batch_results = p_map(
            lambda args: play_single_game(*args[:7], force_white_spec=args[7], force_black_spec=args[8]),
            args_list,
            num_cpus=min(len(args_list), parallelism),
        )

        # Update states and persist PGNs
        for (a, b, w_spec, b_spec), gr in zip(batch, batch_results):
            # Persist PGN
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            res_code = _result_to_code(gr.result)
            meta = f"{gr.white_agent}|{gr.black_agent}|{gr.moves_played}|{gr.final_fen}|{len(gr.pgn_content)}"
            h8 = hashlib.sha256(meta.encode("utf-8")).hexdigest()[:8]
            fname = f"{ts}-g{gr.game_id}-{gr.white_agent}-vs-{gr.black_agent}-{res_code}-{h8}.pgn"
            fpath = pgn_dir / fname
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(gr.pgn_content)
                gr.pgn_path = str(fpath)
            except Exception:
                gr.pgn_path = ""

            results_raw.append(gr)

            # Determine outcome for ratings and stats
            a_state = agent_states[a]
            b_state = agent_states[b]
            # Update color counts and games
            if w_spec == a:
                a_state.as_white += 1
                b_state.as_black += 1
            else:
                b_state.as_white += 1
                a_state.as_black += 1
            a_state.games += 1
            b_state.games += 1

            # Engine metrics aggregation (per-game averages per side)
            if gr.white_agent == a:
                a_state.accuracy_sum += gr.white_accuracy_pct
                a_state.acpl_sum += gr.white_acpl
                b_state.accuracy_sum += gr.black_accuracy_pct
                b_state.acpl_sum += gr.black_acpl
            else:
                b_state.accuracy_sum += gr.white_accuracy_pct
                b_state.acpl_sum += gr.white_acpl
                a_state.accuracy_sum += gr.black_accuracy_pct
                a_state.acpl_sum += gr.black_acpl

            # Illegal attempts aggregation
            if gr.white_agent == a:
                a_state.illegal_attempts_sum += getattr(gr, 'white_illegal_attempts', 0)
                a_state.move_attempts_sum += getattr(gr, 'white_move_attempts', 0)
                b_state.illegal_attempts_sum += getattr(gr, 'black_illegal_attempts', 0)
                b_state.move_attempts_sum += getattr(gr, 'black_move_attempts', 0)
            else:
                b_state.illegal_attempts_sum += getattr(gr, 'white_illegal_attempts', 0)
                b_state.move_attempts_sum += getattr(gr, 'white_move_attempts', 0)
                a_state.illegal_attempts_sum += getattr(gr, 'black_illegal_attempts', 0)
                a_state.move_attempts_sum += getattr(gr, 'black_move_attempts', 0)

            # Update ratings
            a_rating = a_state.rating
            b_rating = b_state.rating
            # Decide winner
            is_draw = "Draw" in gr.result
            white_wins = "White wins" in gr.result
            black_wins = "Black wins" in gr.result
            # Map to r_white, r_black in current game
            r_white = a_rating if w_spec == a else b_rating
            r_black = b_rating if w_spec == a else a_rating
            if is_draw:
                new_white, new_black = trueskill.rate_1vs1(r_white, r_black, drawn=True)
            elif white_wins:
                new_white, new_black = trueskill.rate_1vs1(r_white, r_black)
            else:
                new_black, new_white = trueskill.rate_1vs1(r_black, r_white)
            # Assign back
            if w_spec == a:
                a_state.rating = new_white
                b_state.rating = new_black
            else:
                b_state.rating = new_white
                a_state.rating = new_black

            # W/L/D counters
            if is_draw:
                a_state.draws += 1
                b_state.draws += 1
                a_result_side = "draw"
                b_result_side = "draw"
                winner_spec = None
            else:
                white_side_spec = w_spec
                winner_spec = white_side_spec if white_wins else (b if white_side_spec == a else a)
                if winner_spec == a:
                    a_state.wins += 1
                    b_state.losses += 1
                    a_result_side = "win"
                    b_result_side = "loss"
                else:
                    b_state.wins += 1
                    a_state.losses += 1
                    a_result_side = "loss"
                    b_result_side = "win"

            # H2H update (store ordered pair key)
            key = tuple(sorted([a, b]))
            if key not in h2h:
                h2h[key] = {"a_wins": 0, "b_wins": 0, "draws": 0}
            if is_draw:
                h2h[key]["draws"] += 1
            else:
                if winner_spec == key[0]:
                    h2h[key]["a_wins"] += 1
                else:
                    h2h[key]["b_wins"] += 1

            # Game record for JSON
            games.append({
                "id": gr.game_id,
                "white_agent_spec": gr.white_agent,
                "black_agent_spec": gr.black_agent,
                "result": _result_to_code(gr.result),
                "game_over_reason": gr.game_over_reason,
                "moves_played": gr.moves_played,
                "final_fen": gr.final_fen,
                "engine_metrics": {
                    "white_accuracy_pct": gr.white_accuracy_pct,
                    "black_accuracy_pct": gr.black_accuracy_pct,
                    "white_acpl": gr.white_acpl,
                    "black_acpl": gr.black_acpl,
                },
            "illegal_metrics": {
                "white_illegal_attempts": gr.white_illegal_attempts,
                "black_illegal_attempts": gr.black_illegal_attempts,
                "white_move_attempts": gr.white_move_attempts,
                "black_move_attempts": gr.black_move_attempts,
            },
                "pgn_path": gr.pgn_path,
            })

            # Histories
            agent_states[a].history.append({
                "game_id": gr.game_id,
                "opponent": b,
                "color": "white" if w_spec == a else "black",
                "result": a_result_side,
                "pgn_path": gr.pgn_path,
            })
            agent_states[b].history.append({
                "game_id": gr.game_id,
                "opponent": a,
                "color": "white" if w_spec == b else "black",
                "result": b_result_side,
                "pgn_path": gr.pgn_path,
            })

        total_scheduled += len(batch_results)

        # Show progress display at end of batch
        if tracker:
            console = Console()
            console.print(tracker.render_progress_display())

    # Build standings
    standings = []
    for spec, st in agent_states.items():
        standings.append({
            "agent": spec,
            "rating": {
                "mu": st.rating.mu,
                "sigma": st.rating.sigma,
                "conservative": st.conservative,
            },
            "totals": {
                "games": st.games,
                "as_white": st.as_white,
                "as_black": st.as_black,
                "wins": st.wins,
                "losses": st.losses,
                "draws": st.draws,
            },
            "engine_metrics_avg": {
                "accuracy_pct": (st.accuracy_sum / st.games) if st.games else 0.0,
                "acpl": (st.acpl_sum / st.games) if st.games else 0.0,
            },
            "win_rate": (st.wins + 0.5 * st.draws) / st.games if st.games else 0.0,
            "illegal_metrics": {
                "attempts": st.illegal_attempts_sum,
                "move_attempts": st.move_attempts_sum,
                "illegal_pct": (st.illegal_attempts_sum / st.move_attempts_sum * 100.0) if st.move_attempts_sum else 0.0,
            },
        })
    standings.sort(key=lambda x: x["rating"]["conservative"], reverse=True)

    # Agents section
    agents_json = {}
    for spec, st in agent_states.items():
        agents_json[spec] = {
            "rating": {
                "mu": st.rating.mu,
                "sigma": st.rating.sigma,
                "conservative": st.conservative,
            },
            "totals": {
                "games": st.games,
                "as_white": st.as_white,
                "as_black": st.as_black,
                "wins": st.wins,
                "losses": st.losses,
                "draws": st.draws,
            },
            "engine_metrics_avg": {
                "accuracy_pct": (st.accuracy_sum / st.games) if st.games else 0.0,
                "acpl": (st.acpl_sum / st.games) if st.games else 0.0,
            },
            "illegal_metrics": {
                "attempts": st.illegal_attempts_sum,
                "move_attempts": st.move_attempts_sum,
                "illegal_pct": (st.illegal_attempts_sum / st.move_attempts_sum * 100.0) if st.move_attempts_sum else 0.0,
            },
            "history": st.history,
        }

    # H2H matrix optional
    h2h_matrix = {}
    for (a, b), d in h2h.items():
        h2h_matrix[f"{a}__vs__{b}"] = d

    # Tournament JSON
    summary = {
        "tournament_config": {
            "agents": agent_specs,
            "num_games": num_games,
            "max_games_per_agent": max_games_per_agent,
            "max_moves": max_moves,
            "time_limit": time_limit,
            "parallelism": parallelism,
            "scheduler": scheduler,
            "output_dir": str(out_dir),
            "caps_relaxed": caps_relaxed,
            "games_played": total_scheduled,
        },
        "agents": agents_json,
        "games": games,
        "standings": standings,
        "h2h_matrix": h2h_matrix,
    }

    with open(out_dir / "tournament.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Tournament JSON saved to: {out_dir / 'tournament.json'}")
    print(f"💾 PGNs saved to: {pgn_dir}")

    return summary


@click.command()
@click.option(
    "--agent1",
    default="openai-gpt-4o",
    help="First agent specification. OpenAI: 'openai-gpt-4o', 'openai-gpt-4o-mini', 'openai-gpt-5-mini', 'openai-gpt-5'. Stockfish: 'stockfish-skill1-depth2', 'stockfish-skill5-depth10-time1000'. HF (<10B): 'hf-llama-8b', 'hf-qwen-7b', 'hf-mistral-7b', 'hf-phi-3-mini', 'hf-gemma-7b' or full repo id"
)
@click.option(
    "--agent2", 
    default="stockfish-skill1-depth2",
    help="Second agent specification. OpenAI: 'openai-gpt-4o', 'openai-gpt-4o-mini', 'openai-gpt-5-mini', 'openai-gpt-5'. Stockfish: 'stockfish-skill1-depth2', 'stockfish-skill5-depth10-time1000'. HF (<10B): 'hf-llama-8b', 'hf-qwen-7b', 'hf-mistral-7b', 'hf-phi-3-mini', 'hf-gemma-7b' or full repo id"
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    help="Repeatable agent spec for N-agent tournament (2+ required for tournament mode)"
)
@click.option(
    "--max-moves",
    default=200,
    help="Maximum number of moves per game"
)
@click.option(
    "--time-limit",
    default=15.0,
    help="Time limit per move in seconds"
)
@click.option(
    "--num-games",
    default=1,
    help="Number of games to play"
)
@click.option(
    "--max-games-per-agent",
    default=0,
    help="Soft cap on games per agent (0 = unlimited)"
)
@click.option(
    "--output-dir",
    default="tournament_out",
    help="Directory to store per-game PGNs and tournament.json (tournament mode)"
)
@click.option(
    "--scheduler",
    type=click.Choice(["trueskill", "round_robin"]),
    default="trueskill",
    help="Pairing scheduler for tournament mode"
)
@click.option(
    "--parallelism",
    default=0,
    help="Parallel games per batch (default: min(CPU, remaining games), at least 1)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output for individual games"
)
@click.option(
    "--output",
    default="games.pgn",
    help="Output PGN filename for combined games (duel mode only)"
)
def main(
    agent1: str,
    agent2: str,
    agents: Tuple[str, ...],
    max_moves: int,
    time_limit: float,
    num_games: int,
    max_games_per_agent: int,
    output_dir: str,
    scheduler: str,
    parallelism: int,
    verbose: bool,
    output: str
):
    """Run multiple chess games between configured agents with CLI options."""
    
    # Tournament mode detection
    agents_list = list(agents)
    if len(agents_list) >= 2:
        print("=== N-agent TrueSkill Tournament ===")
        print(f"👥 Agents ({len(agents_list)}): {', '.join(agents_list)}")
        print(f"📊 Target Games: {num_games}")
        print(f"⏱️  Max Moves per Game: {max_moves}")
        print(f"⏰ Time Limit per Move: {time_limit}s")
        print(f"🗂️  Output Dir: {output_dir}")
        print(f"🧮 Scheduler: {scheduler}")
        print(f"🎯 Max Games/Agent (soft): {max_games_per_agent or 'unlimited'}")

        # API key checks for included agent specs
        if any("openai" in a for a in agents_list):
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ Error: OPENAI_API_KEY environment variable not set")
                return
        if any("hf-" in a for a in agents_list):
            if not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")):
                print("❌ Error: HUGGINGFACEHUB_API_TOKEN or HF_TOKEN environment variable not set")
                return

        # Determine parallelism
        cpu_count = os.cpu_count() or 1
        if parallelism and parallelism > 0:
            par = max(1, parallelism)
        else:
            par = max(1, min(cpu_count, num_games))

        summary = run_tournament(
            agent_specs=agents_list,
            num_games=num_games,
            max_moves=max_moves,
            time_limit=time_limit,
            scheduler=scheduler,
            parallelism=par,
            output_dir=output_dir,
            max_games_per_agent=max_games_per_agent,
            verbose=verbose,
        )
        return summary
    else:
        # Backward-compatible duel mode
        print("=== Multi-Game Chess Tournament (Duel Mode) ===")
        print(f"🎮 Agent 1: {agent1}")
        print(f"🎮 Agent 2: {agent2}")
        print(f"📊 Number of Games: {num_games}")
        print(f"⏱️  Max Moves per Game: {max_moves}")
        print(f"⏰ Time Limit per Move: {time_limit}s")
        print(f"🔊 Verbose: {verbose}")
        print(f"💾 Output File: {output}")
        print()

        # Check if OpenAI API key is available if using OpenAI agent
        if "openai" in agent1 or "openai" in agent2:
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ Error: OPENAI_API_KEY environment variable not set")
                print("Please set your OpenAI API key in a .env file or environment variable")
                return
        # Check HF token if using HF agent
        if "hf-" in agent1 or "hf-" in agent2:
            if not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")):
                print("❌ Error: HUGGINGFACEHUB_API_TOKEN or HF_TOKEN environment variable not set")
                print("Please set your Hugging Face token in a .env file or environment variable")
                return

        print(f"🚀 Running {num_games} game{'s' if num_games > 1 else ''}...")

        if num_games > 1 and not verbose:
            progress_tracker = TournamentProgressTracker(num_games, max_moves)
            game_args = [
                (i + 1, agent1, agent2, max_moves, time_limit, verbose, progress_tracker)
                for i in range(num_games)
            ]
            print("\n📊 Progress will be shown during execution...")
            game_results = p_map(
                lambda args: play_single_game(*args),
                game_args,
                num_cpus=min(num_games, os.cpu_count() or 1)
            )
            print("\n" + "=" * 80)
            print("📊 FINAL PROGRESS SUMMARY")
            print("=" * 80)
            for result in game_results:
                if result.result != "ERROR":
                    progress_tracker.mark_game_completed(result.game_id)
                    progress_tracker.update_game_progress(result.game_id, result.moves_played)
            console = Console()
            console.print(progress_tracker.render_progress_display())
        else:
            game_args = [
                (i + 1, agent1, agent2, max_moves, time_limit, verbose)
                for i in range(num_games)
            ]
            game_results = p_map(
                lambda args: play_single_game(*args),
                game_args,
                num_cpus=min(num_games, os.cpu_count() or 1)
            )

        stats = aggregate_results(game_results)
        save_combined_pgn(game_results, output)
        print_summary_stats(stats, game_results)
        return stats


if __name__ == "__main__":
    main()
