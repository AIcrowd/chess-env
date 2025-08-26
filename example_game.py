#!/usr/bin/env python3
"""
Specific gameplay example: OpenAI vs Stockfish Level 1 with Template Customization

This script demonstrates:
- OpenAI agent vs Stockfish Level 1 agent with random color assignment
- Custom prompt template creation and management
- Available template variables and their meanings
- Dynamic template switching during gameplay
- Template customization examples and best practices
- Game runs for up to 200 moves and saves PGN to game.pgn
- Enhanced game termination detection

Key Features Demonstrated:
• Template variable reference guide
• Custom template creation examples
• Dynamic template switching
• Template management tips
• Graceful fallback handling

For comprehensive demonstrations of all features, see example.py
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
    
    # Demonstrate custom prompt template customization
    print("🎨 Customizing OpenAI agent prompt template...")
    
    # Show available template variables
    print("\n📋 Available template variables:")
    print("  • {board_utf}     - Visual board with Unicode pieces")
    print("  • {FEN}           - FEN notation of current position")
    print("  • {side_to_move}  - Which side is to move ('White' or 'Black')")
    print("  • {legal_moves_uci} - Available moves in UCI notation (e.g., 'e2e4')")
    print("  • {legal_moves_san} - Available moves in SAN notation (e.g., 'e4')")
    print("  • {move_history_uci} - Game history in UCI notation")
    print("  • {move_history_san} - Game history in SAN notation")
    print("  • {last_move}     - Description of the last move played")
    
    # Create a custom, focused prompt template
    custom_template = """You are Magnus Carlsen, a chess grandmaster, with deep strategic understanding. Your task is to analyze the current chess position and select the best move available.

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

    print(f"\n🎯 Setting custom template:")
    print("-" * 50)
    print(custom_template)
    print("-" * 50)
    
    # Update the agent with the custom template
    openai_agent.update_prompt_template(custom_template)
    print("✅ Custom prompt template applied!")
    
    # Show examples of other template styles users can create
    print("\n💡 Template customization examples:")
    print("  • Minimal: 'Choose from {legal_moves_uci}. Respond: <uci_move>move</uci_move>'")
    print("  • Position-focused: 'Position: {FEN}\\nMoves: {legal_moves_san}\\nChoose: <uci_move>move</uci_move>'")
    print("  • Strategic: 'Board:\\n{board_utf}\\nYour turn: {side_to_move}\\nLegal moves: {legal_moves_uci}\\nChoose: <uci_move>move</uci_move>'")
    print("  • Historical: 'Game: {move_history_san}\\nCurrent: {side_to_move} to move\\nOptions: {legal_moves_uci}\\nChoose: <uci_move>move</uci_move>'")
    
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
    
    # Demonstrate dynamic template switching
    print("\n🔄 Demonstrating dynamic template switching...")
    
    # Show how to switch to a different template style mid-game
    midgame_template = """Quick tactical analysis needed!

Position: {FEN}
Your turn: {side_to_move}
Legal moves: {legal_moves_uci}

Think tactically and respond with <uci_move>move</uci_move>"""

    print("📝 Switching to midgame template:")
    print("-" * 40)
    print(midgame_template)
    print("-" * 40)
    
    # Update the template (this would normally be done during gameplay)
    openai_agent.update_prompt_template(midgame_template)
    print("✅ Midgame template applied!")
    
    # Play the game
    print("\n🎮 Starting the game with custom template...")
    print("=" * 60)
    
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
    
    # Demonstrate template management
    print("\n🎯 Template Management Tips:")
    print("  • Use update_prompt_template() to change templates anytime")
    print("  • Include only the variables you need in your template")
    print("  • Always wrap moves in <uci_move></uci_move> tags")
    print("  • Test templates with simple examples first")
    print("  • The agent gracefully handles missing template variables")
    
    # Show how to reset to default template
    print("\n🔄 Resetting to default template...")
    openai_agent.update_prompt_template(openai_agent.DEFAULT_PROMPT_TEMPLATE)
    print("✅ Default template restored!")
    
    return result


if __name__ == "__main__":
    main()
