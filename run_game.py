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

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
from p_tqdm import p_map
from rich.console import Console
from rich.json import JSON

from agents import OpenAIAgent, StockfishAgent
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


class AgentFactory:
    """Factory class for creating chess agents from string specifications."""
    
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
            skill_level = 1
            depth = 2
            time_limit_ms = 1000
            
            for part in parts[1:]:
                if part.startswith("skill"):
                    try:
                        skill_level = int(part.split("=")[1])
                    except (IndexError, ValueError):
                        pass
                elif part.startswith("depth"):
                    try:
                        depth = int(part.split("=")[1])
                    except (IndexError, ValueError):
                        pass
                elif part.startswith("time"):
                    try:
                        time_limit_ms = int(part.split("=")[1])
                    except (IndexError, ValueError):
                        pass
            
            return StockfishAgent(
                skill_level=skill_level,
                depth=depth,
                time_limit_ms=time_limit_ms
            )
            
        else:
            raise ValueError(f"Unknown agent type: {agent_spec}")


def play_single_game(
    game_id: int,
    agent1_spec: str,
    agent2_spec: str,
    max_moves: int,
    time_limit: float,
    verbose: bool
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
        
        # Play the game
        result = env.play_game(verbose=verbose)
        
        # Generate PGN content
        pgn_content = env._generate_pgn_content(include_metadata=True)
        
        # Create game result
        game_result = GameResult(
            game_id=game_id,
            result=result['result'],
            moves_played=result['moves_played'],
            white_agent=white_name,
            black_agent=black_name,
            game_over_reason=result['game_over_reason'],
            final_fen=result['final_fen'],
            pgn_content=pgn_content
        )
        
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
            agent_games[agent] += 1
            
            # Count white/black assignments
            if agent == result.white_agent:
                agent_white_games[agent] += 1
            else:
                agent_black_games[agent] += 1
        
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
        "win_rates": {}
    }
    
    # Calculate win rates
    for agent, wins in agent_wins.items():
        games_played = agent_games.get(agent, 0)
        stats["win_rates"][agent] = (wins / games_played * 100) if games_played > 0 else 0
    
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
        print(f"🎮 {agent}:")
        print(f"   Games Played: {games_played}")
        print(f"   As White: {white_games}")
        print(f"   As Black: {black_games}")
        print(f"   Points: {wins}")
        print(f"   Win Rate: {win_rate:.1f}%")
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
            "win_percentage": f"{win_rate:.1f}%"
        }
    
    # Print prettified JSON using rich
    console.print(JSON(json.dumps(json_stats, indent=2)))


@click.command()
@click.option(
    "--agent1",
    default="openai-gpt-4o",
    help="First agent specification. OpenAI: 'openai-gpt-4o', 'openai-gpt-4o-mini', 'openai-gpt-5-mini', 'openai-gpt-5'. Stockfish: 'stockfish-skill1-depth2', 'stockfish-skill5-depth10-time1000'"
)
@click.option(
    "--agent2", 
    default="stockfish-skill1-depth2",
    help="Second agent specification. OpenAI: 'openai-gpt-4o', 'openai-gpt-4o-mini', 'openai-gpt-5-mini', 'openai-gpt-5'. Stockfish: 'stockfish-skill1-depth2', 'stockfish-skill5-depth10-time1000'"
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
    "--verbose",
    is_flag=True,
    help="Enable verbose output for individual games"
)
@click.option(
    "--output",
    default="games.pgn",
    help="Output PGN filename for combined games"
)
def main(
    agent1: str,
    agent2: str,
    max_moves: int,
    time_limit: float,
    num_games: int,
    verbose: bool,
    output: str
):
    """Run multiple chess games between configured agents with CLI options."""
    
    # Check if OpenAI API key is available if using OpenAI agent
    if "openai" in agent1 or "openai" in agent2:
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key in a .env file or environment variable")
            return
    
    print("=== Multi-Game Chess Tournament ===")
    print(f"🎮 Agent 1: {agent1}")
    print(f"🎮 Agent 2: {agent2}")
    print(f"📊 Number of Games: {num_games}")
    print(f"⏱️  Max Moves per Game: {max_moves}")
    print(f"⏰ Time Limit per Move: {time_limit}s")
    print(f"🔊 Verbose: {verbose}")
    print(f"💾 Output File: {output}")
    print()
    
    # Run games with parallel execution (handles both single and multiple games efficiently)
    print(f"🚀 Running {num_games} game{'s' if num_games > 1 else ''}...")
    
    # Prepare arguments for parallel execution
    game_args = [
        (i + 1, agent1, agent2, max_moves, time_limit, verbose)
        for i in range(num_games)
    ]
    
    # Run games in parallel
    game_results = p_map(
        lambda args: play_single_game(*args),
        game_args,
        num_cpus=min(num_games, os.cpu_count() or 1)
    )
    
    # Aggregate results
    stats = aggregate_results(game_results)
    
    # Save combined PGN
    save_combined_pgn(game_results, output)
    
    # Print summary
    print_summary_stats(stats, game_results)
    
    return stats


if __name__ == "__main__":
    main()
