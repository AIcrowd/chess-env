#!/usr/bin/env python3
"""
Comprehensive demonstration of all chess environment features

This script demonstrates ALL available features of the chess environment:
- Basic environment usage and game play
- Multiple games and statistics
- Custom starting positions (FEN)
- PGN export with metadata
- Agent analysis and comparison
- Stockfish agent functionality
- OpenAI agent functionality
- Chess board rendering options
- Enhanced game termination detection
- Clean game display

For a specific gameplay example, see example_game.py
"""

import os

import chess
from agents import (
    ChessAgent,
    FirstMoveAgent,
    LastMoveAgent,
    OpenAIAgent,
    RandomAgent,
    StockfishAgent,
)
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
    
    # Show enhanced termination information if available
    if result['game_over_reason'] != "max_moves":
        print(f"🏁 Termination details: {result['game_over_reason']}")
    
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


def demonstrate_stockfish_agent():
    """Demonstrate the Stockfish agent functionality."""
    print("\n=== Stockfish Agent Demo ===\n")
    
    try:
        # Create a Stockfish agent with default settings
        print("1. Creating Stockfish agent with default settings...")
        stockfish_agent = StockfishAgent()
        print(f"✅ Stockfish agent created successfully")
        print(f"   Binary path: {stockfish_agent.stockfish_path}")
        print(f"   Skill level: {stockfish_agent.skill_level}")
        print(f"   Search depth: {stockfish_agent.depth}")
        print(f"   Hash size: {stockfish_agent.hash_size_mb} MB")
        print(f"   Threads: {stockfish_agent.threads}")
        print()
        
        # Test against a random agent
        print("2. Testing Stockfish vs Random agent...")
        random_agent = RandomAgent()
        
        env = ChessEnvironment(stockfish_agent, random_agent, max_moves=6, time_limit=3.0)
        
        print("Playing a very short game...")
        result = env.play_game(verbose=True)
        print(f"Game result: {result['result']} in {result['moves_played']} moves")
        print()
        
        # Test different skill levels
        print("3. Testing different skill levels...")
        skill_levels = [5, 10, 15, 20]
        
        for skill in skill_levels:
            print(f"Testing skill level {skill}...")
            try:
                # Create new agent with specific skill level
                test_agent = StockfishAgent(skill_level=skill, depth=8)
                print(f"  ✅ Skill {skill}: Agent created successfully")
                test_agent.close()
                
            except Exception as e:
                print(f"  ❌ Skill {skill}: Failed - {e}")
        
        print()
        
        # Test ELO rating limitation
        print("4. Testing ELO rating limitation...")
        try:
            elo_agent = StockfishAgent(elo_rating=1200, depth=8)
            print(f"✅ ELO-limited agent created (1200 rating)")
            print(f"   ELO rating: {elo_agent.elo_rating}")
            elo_agent.close()
            
        except Exception as e:
            print(f"❌ ELO limitation failed: {e}")
        
        print()
        
        # Test custom parameters
        print("5. Testing custom parameters...")
        try:
            custom_agent = StockfishAgent(
                depth=12,
                hash_size_mb=256,
                threads=2,
                time_limit_ms=1000,
                parameters={"Contempt": 10, "Min Split Depth": 2}
            )
            print(f"✅ Custom agent created successfully")
            print(f"   Custom depth: {custom_agent.depth}")
            print(f"   Custom hash: {custom_agent.hash_size_mb} MB")
            print(f"   Custom threads: {custom_agent.threads}")
            print(f"   Time limit: {custom_agent.time_limit_ms} ms")
            custom_agent.close()
            
        except Exception as e:
            print(f"❌ Custom parameters failed: {e}")
        
        print()
        
        # Test parameter updates
        print("6. Testing parameter updates...")
        try:
            update_agent = StockfishAgent(depth=10)
            print(f"✅ Agent created with depth 10")
            
            # Update parameters during runtime
            update_agent.set_depth(15)
            update_agent.set_skill_level(15)
            update_agent.set_time_limit(2000)
            
            print(f"   Updated depth: {update_agent.depth}")
            print(f"   Updated skill: {update_agent.skill_level}")
            print(f"   Updated time limit: {update_agent.time_limit_ms} ms")
            update_agent.close()
            
        except Exception as e:
            print(f"❌ Parameter updates failed: {e}")
        
        # Clean up the main agent
        stockfish_agent.close()
        
        print("\n✅ Stockfish agent demonstration completed successfully!")
        return True
        
    except RuntimeError as e:
        if "Stockfish binary not found" in str(e):
            print("❌ Stockfish not available on this system")
            print("   To use Stockfish agent, please install Stockfish:")
            print("   - macOS: brew install stockfish")
            print("   - Ubuntu/Debian: sudo apt install stockfish")
            print("   - Windows: Download from https://stockfishchess.org/download/")
            print("   - Or set STOCKFISH_PATH environment variable")
            print()
            print("   For now, continuing with other demonstrations...")
            return False
        else:
            print(f"❌ Stockfish agent failed: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error with Stockfish agent: {e}")
        return False


def demonstrate_openai_agent():
    """Demonstrate the OpenAI agent functionality."""
    print("\n=== OpenAI Agent Demo ===\n")
    
    try:
        # Check if OpenAI API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OpenAI API key not available")
            print("   To use OpenAI agent, please set the OPENAI_API_KEY environment variable:")
            print("   export OPENAI_API_KEY='your-api-key-here'")
            print()
            print("   For now, continuing with other demonstrations...")
            return False
        
        print("1. Testing OpenAI API connection...")
        openai_agent = OpenAIAgent(api_key=api_key, model="gpt-5-mini")
        
        if openai_agent.test_connection():
            print("✅ OpenAI API connection successful")
        else:
            print("❌ OpenAI API connection failed")
            return False
        
        print(f"   Model: {openai_agent.model}")
        print(f"   Temperature: {openai_agent.temperature}")
        print(f"   Max tokens: {openai_agent.max_tokens}")
        print()
        
        # Test prompt template functionality
        print("2. Testing prompt template functionality...")
        print("   Current prompt template:")
        print("   " + "-" * 50)
        print(openai_agent.get_prompt_template())
        print("   " + "-" * 50)
        print()
        
        # Test custom prompt template
        print("3. Testing custom prompt template...")
        custom_template = """You are a tactical chess player focused on finding the most effective move in each position.

CURRENT BOARD STATE:
{board_utf}

POSITION ANALYSIS:
- FEN notation: {FEN}
- Side to move: {side_to_move}
- Last move played: {last_move}

MOVES AVAILABLE:
- Legal moves in UCI notation: {legal_moves_uci}
- Legal moves in SAN notation: {legal_moves_san}

GAME CONTEXT:
- Move history in UCI notation: {move_history_uci}
- Move history in SAN notation: {move_history_san}

YOUR TASK:
1. Evaluate the position for:
   - Immediate tactical opportunities
   - Piece development and coordination
   - Control of the center
   - King safety considerations

2. Choose the strongest move from the legal options listed above.

3. CRITICAL: Respond with your move in UCI notation (e.g., "e2e4", "g1f3") wrapped in <uci_move></uci_move> tags.

4. NEVER use SAN notation like "e4" or "Nf3" in your response.

5. If the position is clearly lost, respond with <uci_move>resign</uci_move>

EXAMPLE FORMAT:
- <uci_move>e2e4</uci_move>
- <uci_move>g1f3</uci_move>
- <uci_move>e1g1</uci_move>

Remember: UCI notation only, wrapped in <uci_move></uci_move> tags."""
        
        openai_agent.update_prompt_template(custom_template)
        print("✅ Custom prompt template updated")
        print()
        
        # Test generation parameters
        print("4. Testing generation parameters...")
        openai_agent.update_generation_params(max_completion_tokens=500)
        print("✅ Generation parameters updated")
        print(f"   Current temperature: {openai_agent.temperature}")
        print(f"   New max completion tokens: {openai_agent.max_tokens}")
        print()
        
        # Test fallback behavior configuration
        print("5. Testing fallback behavior...")
        print(f"   Current fallback behavior: {openai_agent.get_fallback_behavior()}")
        
        # Test updating fallback behavior
        openai_agent.update_fallback_behavior("resign")
        print(f"   Updated fallback behavior: {openai_agent.get_fallback_behavior()}")
        
        # Reset to default
        openai_agent.update_fallback_behavior("random_move")
        print(f"   Reset fallback behavior: {openai_agent.get_fallback_behavior()}")
        print()
        
        # Test move selection (without playing a full game to save API calls)
        print("6. Testing move selection...")
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        
        print(f"   Starting position, legal moves: {len(legal_moves)}")
        print(f"   Sample legal moves: {[board.san(move) for move in legal_moves[:5]]}")
        
        # Test prompt formatting
        prompt = openai_agent._format_prompt(board, legal_moves, [], "White")
        print("   Generated prompt preview:")
        print("   " + "-" * 50)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("   " + "-" * 50)
        print()
        
        # Test actual OpenAI API call to see the response
        print("7. Testing OpenAI API response...")
        try:
            print("   Calling OpenAI API with the formatted prompt...")
            print(f"   Prompt length: {len(prompt)} characters")
            print(f"   Generation params: {openai_agent.generation_params}")
            
            response = openai_agent._call_openai_api(prompt)
            print("   ✅ OpenAI API response received:")
            print("   " + "-" * 50)
            print(f"   Response length: {len(response) if response else 0} characters")
            print(f"   Response: '{response}'")
            print("   " + "-" * 50)
            
            # Test move parsing
            print("   Testing move parsing...")
            try:
                move = openai_agent._parse_move(response, legal_moves, board)
                print(f"   ✅ Successfully parsed move: {move.uci()}")
                print(f"   Move in SAN: {board.san(move)}")
            except ValueError as e:
                print(f"   ❌ Move parsing failed: {e}")
                print("   This demonstrates the strict parsing requirements.")
            
        except Exception as e:
            print(f"   ❌ OpenAI API call failed: {e}")
            print("   This might be due to API limits or network issues.")
        
        print()
        
        print("✅ OpenAI agent demonstration completed successfully!")
        print("   Note: This agent requires an OpenAI API key and will make API calls.")
        print("   For production use, consider fine-tuning your own model.")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI agent demonstration failed: {e}")
        return False


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
    
    # Play e4 using UCI notation
    env.board.push(chess.Move.from_uci("e2e4"))
    env.move_history = ["e2e4"]
    print("After 1. e2e4:")
    print(env.display_board(highlight_last_move=True))
    print()
    
    # Play e5 using UCI notation
    env.board.push(chess.Move.from_uci("e7e5"))
    env.move_history = ["e2e4", "e7e5"]
    print("After 1. e2e4 2. e7e5:")
    print(env.display_board(highlight_last_move=True))
    print()
    
    # Show move sequence
    print("5. Move Sequence Display:")
    print("-" * 40)
    
    # Use consolidated move sequence display with professional style and maximum spacing
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3")
    ]
    
    print("Using professional move sequence display with maximum spacing...")
    # Use the consolidated method with professional style and 4-line spacing
    print(env.renderer.render_move_sequence(env.board, moves, style="professional", spacing=4))
    
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


def demonstrate_clean_game():
    """Demonstrate a clean, easy-to-follow game display."""
    print("\n=== Clean Game Demo ===\n")
    
    # Create environment with clean display
    env = ChessEnvironment(RandomAgent(), FirstMoveAgent(), max_moves=10, time_limit=1.0)
    
    print("Playing a clean, easy-to-follow game...")
    print("=" * 60)
    
    # Custom game display function
    def play_clean_game():
        env.reset()
        move_count = 0
        
        # Show initial position
        print("Initial Position:")
        print(env.display_board())
        print()
        
        while not env.is_game_over() and move_count < env.max_moves:
            current_side = env.get_side_to_move()
            current_agent = env.agent1 if current_side == "White" else env.agent2
            
            # Get legal moves (show only first 5 for clarity)
            legal_moves = env.get_legal_moves_uci()
            display_moves = legal_moves[:5]
            if len(legal_moves) > 5:
                display_moves.append(f"... and {len(legal_moves) - 5} more")
            
            print(f"Move {move_count + 1}: {current_side}'s turn")
            print(f"Legal moves: {', '.join(display_moves)}")
            
            # Get and play move
            move = env.play_agent_move(current_agent, current_side)
            if move is None:
                print(f"❌ {current_side} failed to provide a valid move")
                break
            
            # Show the move played
            print(f"✅ {current_side} plays: {move.uci()}")
            
            # Show board after move (only rich version for clarity)
            print(f"\nPosition after {move_count + 1}. {move.uci()}:")
            env.renderer.render_board(env.board, last_move=move, output_mode="clean")
            print()
            
            # Show move summary
            print(f"Move {move_count + 1}: {move.uci()} | Side: {current_side} | Agent: {current_agent.__class__.__name__}")
            print("-" * 60)
            
            move_count += 1
        
        # Game ended
        result = env.get_game_result()
        if result is None:
            result = "Draw (max moves reached)"
        
        print(f"\n🎯 Game Over: {result}")
        print(f"📊 Total moves: {move_count}")
        print(f"📝 Move history: {' '.join(env.move_history)}")
        
        return {
            "result": result,
            "moves_played": move_count,
            "move_history": env.move_history.copy()
        }
    
    # Play the clean game
    result = play_clean_game()
    
    return result


def demonstrate_game_termination():
    """Demonstrate the enhanced game termination detection."""
    print("\n=== Game Termination Detection Demo ===\n")
    
    # Test different termination scenarios
    print("1. Testing checkmate detection...")
    
    # Create a position that leads to quick checkmate
    checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    env = ChessEnvironment(FirstMoveAgent(), FirstMoveAgent(), max_moves=5, initial_fen=checkmate_fen)
    
    print("Starting from Fool's Mate position:")
    print(env.display_board())
    print()
    
    # Play the game
    result = env.play_game(verbose=False)
    print(f"Game result: {result['result']}")
    print(f"Game over reason: {result['game_over_reason']}")
    print(f"Moves played: {result['moves_played']}")
    print()
    
    # Test stalemate detection
    print("2. Testing stalemate detection...")
    stalemate_fen = "k7/8/1K6/8/8/8/8/8 w - - 0 1"
    env2 = ChessEnvironment(FirstMoveAgent(), FirstMoveAgent(), max_moves=3, initial_fen=stalemate_fen)
    
    print("Starting from stalemate position:")
    print(env2.display_board())
    print()
    
    result2 = env2.play_game(verbose=False)
    print(f"Game result: {result2['result']}")
    print(f"Game over reason: {result2['game_over_reason']}")
    print(f"Moves played: {result2['moves_played']}")
    print()
    
    # Test draw by insufficient material
    print("3. Testing draw by insufficient material...")
    draw_fen = "8/8/8/8/8/8/8/4K3 w - - 0 1"
    env3 = ChessEnvironment(FirstMoveAgent(), FirstMoveAgent(), max_moves=2, initial_fen=draw_fen)
    
    print("Starting from King vs King position:")
    print(env3.display_board())
    print()
    
    result3 = env3.play_game(verbose=False)
    print(f"Game result: {result3['result']}")
    print(f"Game over reason: {result3['game_over_reason']}")
    print(f"Moves played: {result3['moves_played']}")
    
    return [result, result2, result3]


def main():
    """Run all demonstrations."""
    print("Chess Environment Demonstrations")
    print("=" * 40)
    
    try:
        # Clean game demo (most important)
        result1 = demonstrate_clean_game()
        
        # Game termination detection demo
        termination_results = demonstrate_game_termination()
        
        # Basic usage
        result2 = demonstrate_basic_usage()
        
        # Multiple games
        results3 = demonstrate_multiple_games()
        
        # Custom positions
        demonstrate_custom_positions()
        
        # FEN initialization
        demonstrate_fen_initialization()

        # PGN export
        demonstrate_pgn_export()
        
        # Agent analysis
        demonstrate_agent_analysis()
        
        # Stockfish agent (if available)
        stockfish_success = demonstrate_stockfish_agent()
        
        # OpenAI agent (if API key available)
        openai_success = demonstrate_openai_agent()
        
        # Chess rendering
        demonstrate_chess_rendering()
        
        print("\n=== All demonstrations completed successfully! ===")
        if not stockfish_success:
            print("Note: Stockfish agent demonstration was skipped due to missing Stockfish binary.")
        if not openai_success:
            print("Note: OpenAI agent demonstration was skipped due to missing API key.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
