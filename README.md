# Chess Environment

A Python-based chess environment for running games between AI agents, built according to the AIcrowd Chess Challenge specifications. Features enhanced game termination detection, flexible prompt templates, and comprehensive PGN export capabilities.

## 🚀 Quick Start

Get up and running in under 2 minutes:

### 1. Setup Environment
```bash
# Create and activate conda environment
conda create python=3.11 --name chess
conda activate chess

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Game
```bash
# Quick game between two random agents
python run_game.py

# Or run a single game programmatically
python -c "
from env import ChessEnvironment, RandomAgent
env = ChessEnvironment(RandomAgent(), RandomAgent())
result = env.play_game(verbose=True)
print(f'Game Result: {result[\"result\"]}')
"
```

### 3. Try Different Agents
```bash
# OpenAI vs Stockfish (requires API key setup)
python run_game.py --agent1 openai-gpt-4o-mini --agent2 stockfish-skill5-depth10

# Multiple games with custom settings
python run_game.py --agent1 stockfish-skill1-depth10 --agent2 stockfish-skill10-depth10 --num-games 5
```

**That's it!** You're now running chess games with AI agents. Continue reading for advanced features and customization.

---

## ✨ Features

- **Two-player chess games** between agent classes
- **Abstract agent interface** for easy implementation of different strategies
- **Multiple agent implementations** including Random, FirstMove, LastMove, Stockfish, and OpenAI agents
- **Modular agent architecture** in separate `agents/` folder for easy extension
- **Enhanced game termination detection** (checkmate, stalemate, insufficient material, fifty-move rule, threefold repetition)
- **Comprehensive game state tracking** including FEN notation, move history, and PGN output
- **Flexible prompt template system** for OpenAI agents with customizable placeholders
- **Advanced PGN export** with metadata, termination reasons, and game statistics
- **Rich chess board rendering** with Unicode pieces and optional Rich CLI styling
- **Configurable game parameters** (max moves, time limits)
- **Built-in validation** and error handling
- **Comprehensive test suite** for safe development and updates

## 📋 Requirements

- Python 3.8+
- `python-chess` - Chess board representation and game logic
- `stockfish` - Stockfish chess engine integration
- `openai` - OpenAI API integration for GPT-based agents
- `rich` - Enhanced terminal rendering with colors and styling (optional, falls back to plain text)

## 🔧 Installation

1. **Create and Activate the chess conda environment:**
   ```bash
   conda create python=3.11 --name chess
   conda activate chess
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables (optional):**
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env with your API keys and configuration
   # See Configuration section below for details
   ```

## ⚙️ Configuration

The chess environment supports configuration through environment variables, which can be set in a `.env` file for convenience.

### Environment Variables

Create a `.env` file in the project root by copying `env.example`:

```bash
cp env.example .env
```

#### OpenAI Agent Configuration

To use the OpenAI agent, you'll need an OpenAI API key:

```bash
# Required for OpenAI agent
OPENAI_API_KEY=your_openai_api_key_here

# Optional OpenAI settings
OPENAI_MODEL=gpt-5                   # Model to use (default: gpt-5)
OPENAI_TEMPERATURE=0.1               # Generation temperature (default: 0.1)
OPENAI_MAX_TOKENS=50                 # Max tokens per response (default: 50)
OPENAI_TIMEOUT=30.0                  # API timeout in seconds (default: 30.0)
OPENAI_FALLBACK_BEHAVIOR=random_move # Fallback when parsing fails (random_move/resign)
```

**Getting an OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Create a new API key
4. Copy the key to your `.env` file

**Model Selection:**
- **`gpt-5`** - Most advanced model, highest capability and cost
- **`gpt-4o-mini`**: Fast and cost-effective
- **`gpt-4o`**: High capability, moderate cost

**Fallback Behavior:**
- **`random_move`** (default): Choose a random legal move if parsing fails
- **`resign`**: Resign the game if no valid move can be parsed

#### Stockfish Configuration

If Stockfish is not in your system PATH, you can specify its location:

```bash
# Optional: Custom Stockfish binary path
STOCKFISH_PATH=/path/to/stockfish
```

**Stockfish Installation:**
- **macOS**: `brew install stockfish`
- **Ubuntu/Debian**: `sudo apt install stockfish`
- **Windows**: Download from [Stockfish website](https://stockfishchess.org/download/)

#### Chess Environment Configuration

```bash
# Optional: Game settings
CHESS_MAX_MOVES=100                  # Maximum moves per game (default: 100)
CHESS_TIME_LIMIT=30.0                # Time limit per move in seconds (default: 30.0)
```

### Example .env File

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef1234567890abcdef

# OpenAI Model Settings
OPENAI_MODEL=gpt-5
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=50
OPENAI_TIMEOUT=30.0
OPENAI_FALLBACK_BEHAVIOR=random_move

# Stockfish Configuration
STOCKFISH_PATH=/opt/homebrew/bin/stockfish

# Chess Environment Settings
CHESS_MAX_MOVES=100
CHESS_TIME_LIMIT=30.0
```

### Configuration Priority

Configuration values are loaded in this order (highest to lowest priority):
1. **Function parameters** (passed directly to agent constructors)
2. **Environment variables** (from `.env` file or system environment)
3. **Default values** (hardcoded in the agent classes)

### Security Notes

- **Never commit your `.env` file** to version control
- **Keep your API keys secure** and don't share them
- **Use environment variables** for production deployments
- **Rotate API keys regularly** for security

## 🎮 Usage

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
print(f"Game over reason: {result['game_over_reason']}")
```

### 🏆 Tournament Mode with `run_game.py`

The `run_game.py` script is your go-to tool for running multiple chess games:

```bash
# Quick start - single game with default agents
python run_game.py

# Multiple games with custom agents
python run_game.py --agent1 stockfish-skill5-depth10 --agent2 openai-gpt-4o --num-games 10

# Stockfish vs Stockfish with different skill levels
python run_game.py --agent1 stockfish-skill1-depth10 --agent2 stockfish-skill10-depth10 --num-games 10 

# Custom game parameters
python run_game.py --max-moves 50 --time-limit 5.0 --num-games 5 --verbose

# Different OpenAI models
python run_game.py --agent1 openai-gpt-5-mini --agent2 openai-gpt-4o-mini --num-games 3

# Custom output file
python run_game.py --output tournament.pgn --num-games 20
```

**Available Agent Types:**
- **Stockfish**: `stockfish-skill{1-20}-depth{1-20}` (e.g., `stockfish-skill10-depth15`)
- **OpenAI**: `openai-gpt-4o`, `openai-gpt-4o-mini`, `openai-gpt-5-mini`, `openai-gpt-5`
- **Hugging Face (<10B)**: aliases for fast setup (requires token), or pass full repo id
  - `hf-llama-8b` → `meta-llama/Meta-Llama-3-8B-Instruct`
  - `hf-llama3-8b` → `meta-llama/Meta-Llama-3-8B-Instruct`
  - `hf-llama-3.1-8b` → `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - `hf-qwen-7b` → `Qwen/Qwen2.5-7B-Instruct`
  - `hf-mistral-7b` → `mistralai/Mistral-7B-Instruct-v0.3`
  - `hf-phi-3-mini` → `microsoft/Phi-3-mini-128k-instruct`
  - `hf-phi-3.5-mini` → `microsoft/Phi-3.5-mini-instruct`
  - `hf-gemma-7b` → `google/gemma-7b-it`
- **Built-in**: `random`, `first-move`, `last-move`

### 🧠 Using the OpenAI Agent
### 🤗 Using the Hugging Face Agent

To use Hugging Face models, set a token and choose an alias (focused on <10B parameter models) or pass the full model repo id. See the [HF Inference Providers list](https://huggingface.co/inference/models?desc_sort=pricingInput) for available models and pricing.

```bash
# Required for HF agent
export HUGGINGFACEHUB_API_TOKEN=your_hf_token
# or
export HF_TOKEN=your_hf_token

# Examples
python run_game.py --agent1 hf-llama-8b --agent2 stockfish-skill1-depth2
python run_game.py --agent1 hf-qwen-7b --agent2 openai-gpt-4o-mini
python run_game.py --agent1 hf-mistral-7b --agent2 hf-phi-3-mini --num-games 3

# Use a full repo id directly
python run_game.py --agent1 hf-meta-llama/Meta-Llama-3.1-8B-Instruct --agent2 stockfish-skill1-depth2
```

Aliases currently supported (focused on <10B class):
- `hf-llama-8b`, `hf-llama3-8b`, `hf-llama-3.1-8b`
- `hf-qwen-7b`
- `hf-mistral-7b`
- `hf-phi-3-mini`, `hf-phi-3.5-mini`
- `hf-gemma-7b`

Note: An additional `hf-deepseek` alias maps to `deepseek-ai/DeepSeek-V3-0324` for convenience, even though it exceeds 10B.


The OpenAI agent requires an API key to be configured (see Configuration section above). 

**Note**: The `run_game.py` script supports specific OpenAI models:
- `openai-gpt-4o` - GPT-4 Omni (most capable)
- `openai-gpt-4o-mini` - GPT-4 Omni Mini (faster, cheaper)
- `openai-gpt-5-mini` - GPT-5 Mini (latest model)
- `openai-gpt-5` - GPT-5 (most advanced)

For other models or custom configurations, use the OpenAIAgent class directly in your code.

```python
from env import ChessEnvironment
from agents import OpenAIAgent, RandomAgent

# Create OpenAI agent (will use gpt-5 by default, or settings from .env file)
openai_agent = OpenAIAgent()

# Create a random opponent
random_agent = RandomAgent()

# Play OpenAI vs Random
env = ChessEnvironment(openai_agent, random_agent, max_moves=30)
result = env.play_game(verbose=True)

print(f"OpenAI Agent result: {result['result']}")
print(f"Game over reason: {result['game_over_reason']}")
print(f"Total moves: {result['moves_played']}")
```

**Customizing the OpenAI Agent:**

```python
# Override default settings
openai_agent = OpenAIAgent(
    model="gpt-4o-mini",       # Use smaller model for development
    temperature=0.0,           # Deterministic play
    max_tokens=20              # Limit response length
)

# Custom prompt template
custom_prompt = """You are a chess expert. Choose the best move.

Position: {FEN}
Moves: {legal_moves_uci}
Your turn: {side_to_move}

Respond with your move in UCI notation wrapped in <uci_move></uci_move> tags."""

openai_agent.update_prompt_template(custom_prompt)
```

### Fallback Behavior Configuration

The OpenAI agent supports configurable fallback behavior when it cannot parse a valid move:

```python
# Use random move fallback (default)
agent = OpenAIAgent(
    api_key="your-key",
    fallback_behavior="random_move"  # Choose random legal move if parsing fails
)

# Use resignation fallback
agent = OpenAIAgent(
    api_key="your-key",
    fallback_behavior="resign"       # Resign game if parsing fails
)

# Update fallback behavior during runtime
agent.update_fallback_behavior("resign")
print(f"Current fallback: {agent.get_fallback_behavior()}")
```

### 🐟 Using the Stockfish Agent

The Stockfish agent provides strong chess play using the Stockfish engine:

```python
from env import ChessEnvironment
from agents import StockfishAgent, RandomAgent

# Create Stockfish agent with custom settings
stockfish_agent = StockfishAgent(
    skill_level=15,        # Skill level 0-20 (15 = strong amateur)
    depth=12,              # Search depth
    time_limit_ms=1000     # 1 second per move
)

# Create opponent
random_agent = RandomAgent()

# Play Stockfish vs Random
env = ChessEnvironment(stockfish_agent, random_agent, max_moves=30)
result = env.play_game(verbose=True)

print(f"Stockfish Agent result: {result['result']}")
```

**Stockfish Configuration Options:**

```python
# ELO-limited play (more human-like)
elo_agent = StockfishAgent(elo_rating=1200)

# Custom engine parameters
custom_agent = StockfishAgent(
    hash_size_mb=256,      # 256MB hash table
    threads=4,             # Use 4 CPU threads
    parameters={            # Custom Stockfish parameters
        "Contempt": 10,
        "Min Split Depth": 2
    }
)
```

### 🎯 Starting from Custom Positions

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

### 📁 Exporting Games to PGN Files

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

# Generate PGN content directly
pgn_content = env._generate_pgn_content(include_metadata=True)
with open("games.pgn", "w") as f:
    f.write(pgn_content)
```

**PGN Export Features:**
- **Automatic file extension**: Adds `.pgn` if not provided
- **Rich metadata**: Includes game result, termination reason, move count, FEN positions
- **Enhanced termination detection**: Specific reasons (checkmate, stalemate, insufficient material, fifty-move rule, threefold repetition)
- **Game statistics**: Move count, initial and final FEN positions, agent names
- **Custom positions**: Preserves initial FEN for non-standard starting positions
- **Standard format**: Compatible with chess analysis software (Lichess, Chess.com, etc.)
- **Error handling**: Returns success/failure status with informative error messages

### 🎨 Visual Chess Board Display

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
- **Clean mode**: Option to avoid duplicate output when using Rich CLI

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

### 🎯 Creating Custom Agents

To create your own chess agent, inherit from the `ChessAgent` abstract base class:

```python
from agents import ChessAgent
import chess
import random

class MyCustomAgent(ChessAgent):
    def choose_move(self, board, legal_moves, move_history, side_to_move):
        # Implement your move selection logic here
        # For example, always choose the first legal move
        return legal_moves[0], "First legal move"
        
        # Or implement a more sophisticated strategy
        # return self.evaluate_position(board, legal_moves), "Strategic move"
```

### Available Agents

The `agents/` package includes several pre-implemented agents:

- **`RandomAgent`**: Chooses moves randomly (baseline implementation)
- **`FirstMoveAgent`**: Always chooses the first legal move
- **`LastMoveAgent`**: Always chooses the last legal move
- **`StockfishAgent`**: Uses the Stockfish chess engine for strong play with configurable skill levels
- **`OpenAIAgent`**: Uses OpenAI's GPT models with flexible prompt templates and UCI move parsing

### Agent Package Structure

```
agents/
├── __init__.py          # Package exports
├── base.py              # Abstract ChessAgent base class
├── random_agent.py      # Random move selection
├── first_move_agent.py  # First move selection
├── last_move_agent.py   # Last move selection
├── stockfish_agent.py   # Stockfish chess engine integration
├── openai_agent.py      # OpenAI GPT model integration
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
- `get_legal_moves_uci()`: Get legal moves in UCI notation
- `get_fen()`: Get current board position in FEN notation
- `get_side_to_move()`: Get whose turn it is
- `get_game_termination_reason()`: Get specific reason for game ending
- `play_move(move, comment)`: Play a specific move with optional comment
- `play_game(verbose)`: Play a complete game
- `get_pgn()`: Get the game in PGN format
- `export_pgn_file(filename, include_metadata)`: Export game to PGN file
- `_generate_pgn_content(include_metadata)`: Generate PGN content with enhanced metadata
- `display_board(highlight_last_move, clean)`: Display chess board using Unicode pieces
- `display_game_state(show_move_history)`: Display complete game state
- `display_position_analysis()`: Display position analysis with material count
- `display_move_sequence(moves, start_fen)`: Display sequence of moves
- `set_renderer_options(show_coordinates, show_move_numbers, empty_square_char, use_rich)`: Configure display options

**Constructor Parameters:**
- `agent1`, `agent2`: The two chess agents to play
- `max_moves`: Maximum number of moves before declaring a draw (default: 200)
- `time_limit`: Time limit per move in seconds (default: 10.0)
- `initial_fen`: Optional FEN string to start the game from (default: standard starting position)

## 🚀 Running the Environment

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

## 🧪 Testing

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
pytest tests/test_openai_agent.py
pytest tests/test_stockfish_agent.py
pytest tests/test_chess_renderer.py
pytest tests/test_integration.py
```

### Run Tests with Verbose Output

```bash
pytest -v
```

## 📁 Project Structure

```
chess/
├── env.py                 # Main chess environment
├── chess_renderer.py      # Chess board renderer with Unicode pieces
├── example.py             # Comprehensive feature demonstration
├── run_game.py            # OpenAI vs Stockfish gameplay example
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── agents/               # Chess agent implementations
│   ├── __init__.py       # Agent package exports
│   ├── base.py           # Abstract ChessAgent base class
│   ├── random_agent.py   # Random move selection agent
│   ├── first_move_agent.py # First move selection agent
│   ├── last_move_agent.py  # Last move selection agent
│   ├── stockfish_agent.py  # Stockfish chess engine integration
│   ├── openai_agent.py     # OpenAI GPT model integration
│   └── template_agent.py # Template for new agents
├── docs/                 # Documentation
│   └── OPENAI_AGENT.md   # OpenAI agent detailed documentation
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_openai_agent.py
│   ├── test_stockfish_agent.py
│   ├── test_chess_renderer.py
│   ├── test_integration.py
│   └── test_new_agents.py
└── chess_env/
    └── SPEC.md           # Technical specification
```

## 🛠️ Development

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

## ♟️ Chess Rules Implementation

The environment uses the `python-chess` library which implements:
- Standard chess rules and move validation
- FEN notation parsing and generation
- PGN format support
- Game termination detection (checkmate, stalemate, draw conditions)

## ⚡ Performance Considerations

- **Move generation** is optimized using python-chess
- **Time limits** are enforced per move
- **Maximum move limits** prevent infinite games
- **Memory usage** is minimal for typical game lengths

## 🔮 Future Enhancements

Based on the SPEC.md, potential future features include:
- Stockfish engine integration for evaluation
- Multiple starting positions
- Tournament mode with multiple agents
- Advanced move analysis and statistics
- Web interface integration
- HuggingFace model integration for LLM-based agents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the AIcrowd Chess Challenge and follows the challenge specifications.

## 🆘 Support

For issues related to:
- **Environment setup**: Check the installation instructions
- **Agent implementation**: Review the abstract base class and examples
- **Testing**: Ensure the chess conda environment is activated
- **Performance**: Check time limits and move count settings
