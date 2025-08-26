# Chess Environment

A Python-based chess environment for running games between AI agents, built according to the AIcrowd Chess Challenge specifications.

## Features

- **Two-player chess games** between agent classes
- **Abstract agent interface** for easy implementation of different strategies
- **Random agent implementation** as a baseline
- **Comprehensive game state tracking** including FEN notation, move history, and PGN output
- **Configurable game parameters** (max moves, time limits)
- **Built-in validation** and error handling
- **Test suite** for safe development and updates

## Installation

1. **Activate the chess conda environment:**
   ```bash
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

### Creating Custom Agents

To create your own chess agent, inherit from the `ChessAgent` abstract base class:

```python
from env import ChessAgent
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

### Environment Methods

The `ChessEnvironment` class provides several useful methods:

- `reset(fen)`: Reset to a new position (default: starting position)
- `get_legal_moves()`: Get all legal moves for current position
- `get_fen()`: Get current board position in FEN notation
- `get_side_to_move()`: Get whose turn it is
- `play_move(move)`: Play a specific move
- `play_game(verbose)`: Play a complete game
- `get_pgn()`: Get the game in PGN format

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
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
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
