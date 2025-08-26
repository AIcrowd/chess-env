#!/usr/bin/env python3
"""
Example script demonstrating the chess environment usage.
"""

import chess
from agents import ChessAgent, FirstMoveAgent, LastMoveAgent, RandomAgent
from chess_renderer import RICH_AVAILABLE
from env import ChessEnvironment


def demonstrate_basic_usage():
    """Demonstrate basic environment usage."""
    print("=== Basic Chess Environment Demo ===\n")
    
    # Create agents
    random_agent = RandomAgent()
    first_move_agent = FirstMoveAgent()
    
    # Create environment
    env = ChessEnvironment(random_agent, first_move_agent, max_moves=30, time_limit=2.0)
    
    print(f"White: {random_agent.__class__.__name__}")
    print(f"Black: {first_move_agent.__class__.__name__}")
    print(f"Max moves: {env.max_moves}")
    print(f"Time limit per move: {env.time_limit}s")
    print()
    
    # Play a game
    print("Playing a game...")
    result = env.play_game(verbose=True)
    
    print(f"\n=== Game Results ===")
    print(f"Result: {result['result']}")
    print(f"Moves played: {result['moves_played']}")
    print(f"Game over reason: {result['game_over_reason']}")
    
    return result


def demonstrate_multiple_games():
    """Demonstrate playing multiple games."""
    print("\n=== Multiple Games Demo ===\n")
    
    # Create environment with two random agents
    env = ChessEnvironment(RandomAgent(), RandomAgent(), max_moves=20)
    
    results = []
    for i in range(3):
        print(f"Playing game {i+1}...")
        result = env.play_game(verbose=False)
        results.append(result)
        print(f"  Result: {result['result']}, Moves: {result['moves_played']}")
    
    print(f"\nSummary:")
    print(f"Games played: {len(results)}")
    print(f"Average moves per game: {sum(r['moves_played'] for r in results) / len(results):.1f}")
    
    return results


def demonstrate_custom_positions():
    """Demonstrate playing from custom starting positions."""
    print("\n=== Custom Positions Demo ===\n")
    
    # Test different starting positions
    positions = [
        ("Starting position", chess.STARTING_FEN),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("King and pawn endgame", "8/8/8/8/8/8/4P3/4K3 w - - 0 1"),
    ]
    
    for name, fen in positions:
        print(f"Playing from: {name}")
        print(f"  FEN: {fen}")
        
        # Create environment directly with custom FEN
        env = ChessEnvironment(RandomAgent(), RandomAgent(), max_moves=15, initial_fen=fen)
        print(f"  Side to move: {env.get_side_to_move()}")
        print(f"  Legal moves: {len(env.get_legal_moves())}")
        
        result = env.play_game(verbose=False)
        print(f"  Result: {result['result']}, Moves: {result['moves_played']}")
        print()


def demonstrate_agent_analysis():
    """Demonstrate analyzing agent behavior."""
    print("\n=== Agent Analysis Demo ===\n")
    
    # Create different agent types
    agents = {
        "Random": RandomAgent(),
        "First Move": FirstMoveAgent(),
        "Last Move": LastMoveAgent(),
    }
    
    # Test each agent against a random opponent
    random_opponent = RandomAgent()
    
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name} agent...")
        
        env = ChessEnvironment(agent, random_opponent, max_moves=25)
        
        # Play multiple games to get statistics
        wins = 0
        total_moves = 0
        games_played = 5
        
        for _ in range(games_played):
            result = env.play_game(verbose=False)
            if result['result'] == "White wins":
                wins += 1
            total_moves += result['moves_played']
        
        win_rate = wins / games_played
        avg_moves = total_moves / games_played
        
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Average moves per game: {avg_moves:.1f}")
        print()


def demonstrate_fen_initialization():
    """Demonstrate the new initial_fen parameter functionality."""
    print("\n=== FEN Initialization Demo ===\n")
    
    # Test various interesting positions
    positions = [
        ("Fool's Mate Position", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
        ("Scholar's Mate Setup", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3"),
        ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Endgame: King vs King", "8/8/8/8/8/8/4K3/4k3 w - - 0 1"),
    ]
    
    for name, fen in positions:
        print(f"Creating environment with: {name}")
        print(f"  FEN: {fen}")
        
        # Create environment with custom FEN
        env = ChessEnvironment(RandomAgent(), RandomAgent(), max_moves=20, initial_fen=fen)
        
        print(f"  Initial side to move: {env.get_side_to_move()}")
        print(f"  Legal moves available: {len(env.get_legal_moves())}")
        
        # Play a quick game from this position
        result = env.play_game(verbose=False)
        print(f"  Game result: {result['result']} in {result['moves_played']} moves")
        print()


def demonstrate_pgn_export():
    """Demonstrate the new PGN export functionality."""
    print("\n=== PGN Export Demo ===\n")
    
    # Create a simple game
    env = ChessEnvironment(RandomAgent(), FirstMoveAgent(), max_moves=10)
    
    # Play a short game
    print("Playing a short game for PGN export...")
    result = env.play_game(verbose=False)
    print(f"Game completed: {result['result']} in {result['moves_played']} moves")
    
    # Export to PGN file
    filename = "demo_game"
    print(f"\nExporting game to {filename}.pgn...")
    
    success = env.export_pgn_file(filename)
    if success:
        print(f"✅ Successfully exported to {filename}.pgn")
        
        # Show the PGN content
        print("\nPGN Content:")
        print("-" * 40)
        with open(f"{filename}.pgn", 'r') as f:
            content = f.read()
            print(content)
        print("-" * 40)
        
        # Clean up
        import os
        os.remove(f"{filename}.pgn")
        print(f"🗑️  Cleaned up {filename}.pgn")
    else:
        print("❌ Failed to export PGN file")
    
    # Test export with custom position
    print(f"\nTesting PGN export with custom starting position...")
    custom_env = ChessEnvironment(
        RandomAgent(), 
        FirstMoveAgent(), 
        max_moves=5,
        initial_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    
    # Play a few moves
    custom_env.play_game(verbose=False)
    
    # Export with metadata
    custom_filename = "custom_position_game"
    success = custom_env.export_pgn_file(custom_filename, include_metadata=True)
    
    if success:
        print(f"✅ Successfully exported custom position game to {custom_filename}.pgn")
        
        # Show metadata
        with open(f"{custom_filename}.pgn", 'r') as f:
            content = f.read()
            if '[InitialFEN' in content and '[FinalFEN' in content:
                print("✅ PGN includes custom position metadata")
        
        # Clean up
        import os
        os.remove(f"{custom_filename}.pgn")
        print(f"🗑️  Cleaned up {custom_filename}.pgn")
    else:
        print("❌ Failed to export custom position PGN file")


def demonstrate_chess_rendering():
    """Demonstrate the new chess board rendering functionality."""
    print("\n=== Chess Board Rendering Demo ===\n")
    
    # Create environment
    env = ChessEnvironment(RandomAgent(), FirstMoveAgent(), max_moves=5)
    
    print("1. Basic Board Display:")
    print("-" * 40)
    print(env.display_board())
    print()
    
    print("2. Game State Display:")
    print("-" * 40)
    print(env.display_game_state())
    print()
    
    print("3. Position Analysis:")
    print("-" * 40)
    print(env.display_position_analysis())
    print()
    
    # Play a few moves to show dynamic rendering
    print("4. Playing Some Moves:")
    print("-" * 40)
    
    # Play e4
    env.board.push_san("e4")
    env.move_history = ["e4"]
    print("After 1. e4:")
    print(env.display_board(highlight_last_move=True))
    print()
    
    # Play e5
    env.board.push_san("e5")
    env.move_history = ["e4", "e5"]
    print("After 1. e4 2. e5:")
    print(env.display_board(highlight_last_move=True))
    print()
    
    # Show move sequence
    print("5. Move Sequence Display:")
    print("-" * 40)
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3")
    ]
    print(env.display_move_sequence(moves))
    
    # Test renderer options
    print("6. Renderer Options:")
    print("-" * 40)
    
    print("Without coordinates:")
    env.set_renderer_options(show_coordinates=False)
    print(env.display_board())
    print()
    
    print("With move numbers:")
    env.set_renderer_options(show_coordinates=True, show_move_numbers=True)
    print(env.renderer.render_board(env.board, move_number=3))
    print()
    
    print("With different empty square characters:")
    env.set_renderer_options(show_coordinates=True, show_move_numbers=False)
    
    # Test different empty square characters
    for char in ["·", ".", "-", " "]:
        env.set_renderer_options(empty_square_char=char)
        print(f"Empty squares as '{char}':")
        print(env.display_board())
        print()
    
    # Test rich vs plain rendering
    print("Rich CLI rendering (if available):")
    env.set_renderer_options(use_rich=True, empty_square_char="·")
    rich_output = env.display_board()
    print(rich_output)
    print()
    
    print("Plain text rendering:")
    env.set_renderer_options(use_rich=False, empty_square_char="·")
    plain_output = env.display_board()
    print(plain_output)
    print()
    
    # Reset to defaults
    env.set_renderer_options(show_coordinates=True, show_move_numbers=False, 
                           empty_square_char="·", use_rich=True)
    
    # Test custom position rendering
    print("7. Custom Position Rendering:")
    print("-" * 40)
    custom_env = ChessEnvironment(
        RandomAgent(), 
        FirstMoveAgent(), 
        initial_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    print("Position after 1. e4:")
    print(custom_env.display_board())
    print()
    
    print("Position analysis for this position:")
    print(custom_env.display_position_analysis())


def main():
    """Run all demonstrations."""
    print("Chess Environment Demonstrations")
    print("=" * 40)
    
    try:
        # Basic usage
        result1 = demonstrate_basic_usage()
        
        # Multiple games
        results2 = demonstrate_multiple_games()
        
        # Custom positions
        demonstrate_custom_positions()
        
        # FEN initialization
        demonstrate_fen_initialization()

        # PGN export
        demonstrate_pgn_export()
        
        # Agent analysis
        demonstrate_agent_analysis()
        
        # Chess rendering
        demonstrate_chess_rendering()
        
        print("\n=== All demonstrations completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
