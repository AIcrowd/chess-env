#!/usr/bin/env python3
"""
Specific gameplay example: OpenAI vs Stockfish Level 1 with Template Customization

This script demonstrates:
- OpenAI agent vs Stockfish Level 1 agent with random color assignment
- Game runs for up to 200 moves and saves PGN to game.pgn
- Enhanced game termination detection

Key Features Demonstrated:
• Template management tips
• Graceful fallback handling
"""

import os
import random

from agents import OpenAIAgent, StockfishAgent
from env import ChessEnvironment


def main():
    """Run a game between OpenAI agent and Stockfish level 1 agent."""
    print("=== OpenAI vs Stockfish Level 1 Game ===\n")
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    # Create agents
    print("Creating agents...")
    openai_agent = OpenAIAgent(
        model="gpt-4o",  # Use GPT-4o
        max_completion_tokens=500  
    )
        
    stockfish_agent = StockfishAgent(
        skill_level=1,        # Level 1 (very weak)
        depth=2,              # Low depth for faster play
        time_limit_ms=1000    # 1 second per move
    )
    
    print(f"\n✅ OpenAI Agent: {openai_agent.__class__.__name__} (with custom template)")
    print(f"✅ Stockfish Agent: {stockfish_agent.__class__.__name__} (Level 1)")
    print()
    
    # Randomly assign White and Black
    agents = [openai_agent, stockfish_agent]
    random.shuffle(agents)
    white_agent, black_agent = agents
    
    print("🎲 Randomly assigning colors...")
    print(f"🏆 White: {white_agent.__class__.__name__}")
    print(f"🏆 Black: {black_agent.__class__.__name__}")
    print()
    
    # Create environment
    print("Setting up chess environment...")
    env = ChessEnvironment(
        agent1=white_agent,       # White (randomly chosen)
        agent2=black_agent,       # Black (randomly chosen)
        max_moves=200,            # Maximum 200 moves
        time_limit=15.0           # 15 seconds per move
    )
    
    print(f"Max moves: {env.max_moves}")
    print(f"Time limit per move: {env.time_limit}s")
    print()
        
    # Play the game
    result = env.play_game(verbose=True)
    
    print("=" * 60)
    print(f"\n🎯 Game Over: {result['result']}")
    print(f"📊 Total moves: {result['moves_played']}")
    print(f"🏆 White: {result['white_agent']}")
    print(f"🏆 Black: {result['black_agent']}")
    print(f"📝 Game over reason: {result['game_over_reason']}")
    
    # Show game over reason
    if result['game_over_reason'] != "max_moves":
        print(f"🏁 Game over reason: {result['game_over_reason']}")
    
    # Show which agent won
    if "wins" in result['result']:
        winner = "White" if "White wins" in result['result'] else "Black"
        winner_agent = result['white_agent'] if winner == "White" else result['black_agent']
        print(f"🏆 Winner: {winner} ({winner_agent})")
    elif "Draw" in result['result']:
        print("🤝 Game ended in a draw")
    
    # Save PGN file
    print("\n💾 Saving PGN file...")
    pgn_content = env._generate_pgn_content(include_metadata=True)
    
    with open("game.pgn", "w", encoding="utf-8") as f:
        f.write(pgn_content)
    
    print("✅ PGN file saved to: game.pgn")
    
    # Show final position
    print(f"\n📋 Final position (FEN): {result['final_fen']}")
    
    return result


if __name__ == "__main__":
    main()
