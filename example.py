#!/usr/bin/env python3
"""
Example script demonstrating the chess environment usage.
"""

import chess
from env import ChessAgent, ChessEnvironment, RandomAgent


class FirstMoveAgent(ChessAgent):
    """Agent that always chooses the first legal move."""
    
    def choose_move(self, board, legal_moves, move_history, side_to_move):
        return legal_moves[0]


class LastMoveAgent(ChessAgent):
    """Agent that always chooses the last legal move."""
    
    def choose_move(self, board, legal_moves, move_history, side_to_move):
        return legal_moves[-1]


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
        
        # Agent analysis
        demonstrate_agent_analysis()
        
        print("\n=== All demonstrations completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
