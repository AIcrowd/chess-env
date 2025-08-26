# Chess Environment

A Python-based chess environment for running games between AI agents, built according to the AIcrowd Chess Challenge specifications.

## Features

- **Two-player chess games** between agent classes
- **Abstract agent interface** for easy implementation of different strategies
- **Random agent implementation** as a baseline
- **Modular agent architecture** in separate `agents/` folder for easy extension
- **Comprehensive game state tracking** including FEN notation, move history, and PGN output
- **Configurable game parameters** (max moves, time limits)
- **Built-in validation** and error handling
- **Test suite** for safe development and updates

## Requirements

- Python 3.8+
- `python-chess` - Chess board representation and game logic
- `Chessnut` - Chess engine integration
- `rich` - Enhanced terminal rendering with colors and styling (optional, falls back to plain text)

## Installation

1. **Create and Activate the chess conda environment:**
   ```bash
   conda create python=3.11 --name chess
   conda activate chess
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Example

```python
from env import ChessEnvironment, RandomAgent

# Create two random agents
agent1 = RandomAgent()
agent2 = RandomAgent()

# Create environment
env = ChessEnvironment(agent1, agent2, max_moves=100, time_limit=5.0)

# Play a game
result = env.play_game(verbose=True)

# Get game results
print(f"Result: {result['result']}")
print(f"Moves played: {result['moves_played']}")
```

### Starting from Custom Positions

You can start games from specific chess positions using FEN notation:

```python
# Start from a midgame position
midgame_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
env = ChessEnvironment(agent1, agent2, initial_fen=midgame_fen)

# Start from an endgame position
endgame_fen = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"
env = ChessEnvironment(agent1, agent2, initial_fen=endgame_fen)

# Play the game from the custom position
result = env.play_game(verbose=True)
```

### Exporting Games to PGN Files

You can export completed games to PGN (Portable Game Notation) files for analysis or sharing:

```python
# Play a game
result = env.play_game(verbose=False)

# Export to PGN file (automatically adds .pgn extension)
success = env.export_pgn_file("my_game")

# Export with custom metadata
success = env.export_pgn_file("tournament_game", include_metadata=True)

# Export without metadata (minimal PGN)
success = env.export_pgn_file("simple_game", include_metadata=False)
```

**PGN Export Features:**
- **Automatic file extension**: Adds `.pgn` if not provided
- **Rich metadata**: Includes game result, termination reason, move count, FEN positions
- **Custom positions**: Preserves initial FEN for non-standard starting positions
- **Standard format**: Compatible with chess analysis software (Lichess, Chess.com, etc.)
- **Error handling**: Returns success/failure status with informative error messages

### Visual Chess Board Display

The environment includes a powerful text-based chess board renderer using Unicode chess pieces:

```python
# Display the current board
print(env.display_board())

# Display with last move highlighted
print(env.display_board(highlight_last_move=True))

# Display complete game state
print(env.display_game_state())

# Display position analysis
print(env.display_position_analysis())

# Display a sequence of moves
moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
print(env.display_move_sequence(moves))

# Configure renderer options
env.set_renderer_options(
    show_coordinates=True,      # Show file/rank coordinates
    show_move_numbers=False,    # Hide move numbers
    empty_square_char="·",      # Use dots for empty squares
    use_rich=True               # Enable rich CLI styling
)
```

**Rendering Features:**
- **Unicode chess pieces**: Beautiful, readable piece symbols (♔♕♖♗♘♙ for White, ♚♛♜♝♞♟ for Black)
- **Empty square visualization**: Clear representation of empty squares using configurable characters (·, ., -, etc.)
- **Coordinate system**: File (a-h) and rank (1-8) coordinates for easy navigation
- **Move highlighting**: Last move is highlighted with brackets `[♙]`
- **Configurable display**: Toggle coordinates, move numbers, empty square characters, and other options
- **Rich CLI support**: Enhanced rendering with colors, alternating square backgrounds, and professional styling
- **Position analysis**: Material count, legal moves, and sample moves
- **Move sequences**: Step-by-step visualization of move sequences
- **Custom positions**: Works with any FEN position

**Rendering Configuration:**
- **Empty Square Characters**: Choose from `·` (dot), `.` (period), `-` (dash), ` ` (space), or any custom character
- **Rich CLI Styling**: Enhanced colors, alternating square backgrounds, and professional appearance
- **Fallback Support**: Automatically falls back to plain text if rich CLI is not available
- **Performance**: Rich rendering only when requested, plain text for maximum compatibility

**Example Output:**
```
   a b c d e f g h  
  +---------------+
8 |♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜| 8
7 |♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟| 7
6 |                | 6
5 |                | 5
4 |                | 4
3 |                | 3
2 |♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙| 2
1 |♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖| 1
  +---------------+
   a b c d e f g h  
```

### Creating Custom Agents

To create your own chess agent, inherit from the `ChessAgent` abstract base class:

```python
from agents import ChessAgent
import chess
import random

class MyCustomAgent(ChessAgent):
    def choose_move(self, board, legal_moves, move_history, side_to_move):
        # Implement your move selection logic here
        # For example, always choose the first legal move
        return legal_moves[0]
        
        # Or implement a more sophisticated strategy
        # return self.evaluate_position(board, legal_moves)
```

### Available Agents

The `agents/` package includes several pre-implemented agents:

- **`RandomAgent`**: Chooses moves randomly (baseline implementation)
- **`FirstMoveAgent`**: Always chooses the first legal move
- **`LastMoveAgent`**: Always chooses the last legal move

### Agent Package Structure

```
agents/
├── __init__.py          # Package exports
├── base.py              # Abstract ChessAgent base class
├── random_agent.py      # Random move selection
├── first_move_agent.py  # First move selection
├── last_move_agent.py   # Last move selection
├── template_agent.py    # Template for new agents
└── ...                  # Future agent implementations
```

### Adding New Agent Types

1. **Create a new file** in the `agents/` folder (e.g., `my_agent.py`)
2. **Inherit from `ChessAgent`** and implement the `choose_move` method
3. **Add to `agents/__init__.py`** to make it available for import
4. **Write tests** in the `tests/` folder
5. **Update documentation** as needed

Example of a new agent:

```python
# agents/my_agent.py
from .base import ChessAgent

class MyAgent(ChessAgent):
    def choose_move(self, board, legal_moves, move_history, side_to_move):
        # Your logic here
        return legal_moves[0]  # Example implementation
```

Then add to `agents/__init__.py`:
```python
from .my_agent import MyAgent
__all__ = [..., "MyAgent"]
```

### Environment Methods

The `ChessEnvironment` class provides several useful methods:

- `reset(fen)`: Reset to a new position (default: starting position)
- `get_legal_moves()`: Get all legal moves for current position
- `get_fen()`: Get current board position in FEN notation
- `get_side_to_move()`: Get whose turn it is
- `play_move(move)`: Play a specific move
- `play_game(verbose)`: Play a complete game
- `get_pgn()`: Get the game in PGN format
- `export_pgn_file(filename, include_metadata)`: Export game to PGN file
- `display_board(highlight_last_move)`: Display chess board using Unicode pieces
- `display_game_state(show_move_history)`: Display complete game state
- `display_position_analysis()`: Display position analysis with material count
- `display_move_sequence(moves, start_fen)`: Display sequence of moves
- `set_renderer_options(show_coordinates, show_move_numbers)`: Configure display options

**Constructor Parameters:**
- `agent1`, `agent2`: The two chess agents to play
- `max_moves`: Maximum number of moves before declaring a draw (default: 200)
- `time_limit`: Time limit per move in seconds (default: 10.0)
- `initial_fen`: Optional FEN string to start the game from (default: standard starting position)

## Running the Environment

### Interactive Mode

```bash
python env.py
```

This will run a sample game between two random agents.

### Programmatic Usage

```python
from env import ChessEnvironment, RandomAgent

# Create environment and play multiple games
env = ChessEnvironment(RandomAgent(), RandomAgent())

for i in range(5):
    print(f"\n=== Game {i+1} ===")
    result = env.play_game(verbose=False)
    print(f"Result: {result['result']}, Moves: {result['moves_played']}")
```

## Testing

The project includes a comprehensive test suite to ensure code quality and prevent regressions.

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=env --cov-report=html
```

### Run Specific Test Files

```bash
pytest tests/test_environment.py
pytest tests/test_agents.py
```

### Run Tests with Verbose Output

```bash
pytest -v
```

## Project Structure

```
chess/
├── env.py                 # Main chess environment
├── chess_renderer.py      # Chess board renderer with Unicode pieces
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── agents/               # Chess agent implementations
│   ├── __init__.py       # Agent package exports
│   ├── base.py           # Abstract ChessAgent base class
│   ├── random_agent.py   # Random move selection agent
│   ├── first_move_agent.py # First move selection agent
│   ├── last_move_agent.py  # Last move selection agent
│   └── template_agent.py # Template for new agents
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_new_agents.py
│   ├── test_chess_renderer.py
│   └── test_integration.py
└── chess_env/
    └── SPEC.md           # Technical specification
```

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

### Pre-commit Checks

```bash
# Format code
black env.py tests/

# Lint code
flake8 env.py tests/

# Type check
mypy env.py tests/
```

### Adding New Features

1. **Write tests first** (TDD approach)
2. **Implement the feature** in the main code
3. **Run tests** to ensure everything works
4. **Update documentation** as needed

### Testing Guidelines

- **Unit tests** for individual methods and classes
- **Integration tests** for complete game scenarios
- **Edge case testing** for error conditions
- **Performance testing** for time-sensitive operations

## Chess Rules Implementation

The environment uses the `python-chess` library which implements:
- Standard chess rules and move validation
- FEN notation parsing and generation
- PGN format support
- Game termination detection (checkmate, stalemate, draw conditions)

## Performance Considerations

- **Move generation** is optimized using python-chess
- **Time limits** are enforced per move
- **Maximum move limits** prevent infinite games
- **Memory usage** is minimal for typical game lengths

## Future Enhancements

Based on the SPEC.md, potential future features include:
- Stockfish engine integration for evaluation
- Multiple starting positions
- Tournament mode with multiple agents
- Advanced move analysis and statistics
- Web interface integration
- HuggingFace model integration for LLM-based agents

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the AIcrowd Chess Challenge and follows the challenge specifications.

## Support

For issues related to:
- **Environment setup**: Check the installation instructions
- **Agent implementation**: Review the abstract base class and examples
- **Testing**: Ensure the chess conda environment is activated
- **Performance**: Check time limits and move count settings
