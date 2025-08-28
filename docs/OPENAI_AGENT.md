# OpenAI Chess Agent

The `OpenAIAgent` is a chess-playing agent that uses OpenAI's API to make chess moves. It follows the SPEC requirements for prompt templates, move parsing, and integration with the chess environment.

## Features

- **OpenAI API Integration**: Uses OpenAI's GPT models for chess move selection
- **Flexible Prompt Templates**: Highly customizable prompts with optional placeholders
- **Robust Move Parsing**: Strict UCI move parsing with <uci_move></uci_move> tags
- **Graceful Fallback Handling**: Intelligent fallback when API calls fail or parsing errors occur
- **Parameter Configuration**: Configurable generation parameters (temperature, max_tokens, etc.)
- **Error Recovery**: Falls back to legal moves or resignation based on user preference
- **Template Validation**: Flexible validation that allows custom templates without strict requirements

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Basic Usage

```python
from agents import OpenAIAgent
import chess

# Create an OpenAI agent
agent = OpenAIAgent(
    api_key="your-api-key",  # Or use environment variable
    model="gpt-5-mini",      # OpenAI model to use
    temperature=0.1,         # Generation temperature
    max_tokens=50            # Maximum tokens to generate
)

# Create a chess board
board = chess.Board()
legal_moves = list(board.legal_moves)

# Get a move from the agent
move = agent.choose_move(board, legal_moves, [], "White")
print(f"Agent chose: {board.san(move)}")
```

## Configuration Options

### Model Selection
```python
# Use different OpenAI models
agent = OpenAIAgent(model="gpt-3.5-turbo")  # Faster, cheaper
agent = OpenAIAgent(model="gpt-5-mini")     # Balanced performance and cost
agent = OpenAIAgent(model="gpt-4")           # Most capable, expensive
```

### Generation Parameters
```python
agent = OpenAIAgent(
    temperature=0.0,    # Deterministic (0.0) or random (1.0)
    max_tokens=100,     # Maximum response length
    top_p=0.9,         # Nucleus sampling
    frequency_penalty=0.1,  # Reduce repetition
    presence_penalty=0.1    # Encourage new topics
)
```

### API Configuration
```python
agent = OpenAIAgent(
    timeout=60.0,           # API call timeout in seconds
    retry_attempts=3,       # Number of retry attempts
    retry_delay=2.0         # Delay between retries
)
```

## Prompt Templates

The agent uses highly flexible prompt templates with placeholders that get filled with actual game data. Unlike the previous strict implementation, you can now create custom templates using only the variables you need.

### Default Template
```python
DEFAULT_PROMPT_TEMPLATE = """You are Magnus Carlsen, a chess grandmaster, with deep strategic understanding. Your task is to analyze the current chess position and select the best move available.

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

5. If you cannot find a good move or believe the position is lost, respond with <uci_move>resign</uci_move>"""
```

### Custom Templates
You can create custom templates with any combination of available placeholders:

```python
# Minimal template (only legal moves)
minimal_template = "Choose the best move from: {legal_moves_uci}. Respond with <uci_move>move</uci_move>"

# Position-focused template
position_template = """Position: {FEN}
Your turn: {side_to_move}
Legal moves: {legal_moves_san}
Choose: <uci_move>move</uci_move>"""

# Strategic template
strategic_template = """Board:
{board_utf}
Your turn: {side_to_move}
Legal moves: {legal_moves_uci}
Choose: <uci_move>move</uci_move>"""

# Historical template
historical_template = """Game: {move_history_san}
Current: {side_to_move} to move
Options: {legal_moves_uci}
Choose: <uci_move>move</uci_move>"""

agent.update_prompt_template(minimal_template)
```

### Available Placeholders
- `{board_utf}`: Visual board representation with Unicode chess pieces
- `{FEN}`: Current board position in Forsyth-Edwards Notation
- `{side_to_move}`: Which side is to move ("White" or "Black")
- `{legal_moves_uci}`: Available moves in UCI notation (e.g., "e2e4", "g1f3")
- `{legal_moves_san}`: Available moves in SAN notation (e.g., "e4", "Nf3")
- `{move_history_uci}`: Game history in UCI notation
- `{move_history_san}`: Game history in SAN notation
- `{last_move}`: Description of the last move played

### Template Flexibility
- **No Required Placeholders**: Include only the variables you need
- **Custom Styles**: Create minimal, focused, or comprehensive templates
- **Dynamic Switching**: Change templates during gameplay
- **Graceful Fallback**: Handles missing variables automatically

## Move Parsing

The agent uses strict UCI (Universal Chess Interface) move parsing to ensure reliable and consistent move extraction:

### Required Format
- **UCI Notation Only**: Moves must be in UCI format (e.g., "e2e4", "g1f3", "e1g1")
- **Tagged Response**: Moves must be wrapped in `<uci_move></uci_move>` tags
- **No SAN Support**: Standard Algebraic Notation (e.g., "e4", "Nf3") is not supported
- **Exact Match**: The move must exactly match one of the legal moves available

### Example Responses
```
✅ Correct: <uci_move>e2e4</uci_move>
✅ Correct: <uci_move>g1f3</uci_move>
✅ Correct: <uci_move>e1g1</uci_move> (kingside castling)
✅ Correct: <uci_move>resign</uci_move>
❌ Incorrect: e4 (missing tags and wrong notation)
❌ Incorrect: <uci_move>Nf3</uci_move> (SAN instead of UCI)
❌ Incorrect: <uci_move>invalid</uci_move> (not a legal move)
```

### Fallback Behavior
The agent has configurable fallback behavior when parsing fails:

```python
# Configure fallback behavior
agent = OpenAIAgent(fallback_behavior="random_move")  # Choose random legal move
agent = OpenAIAgent(fallback_behavior="resign")       # Resign the game

# Update fallback behavior during runtime
agent.update_fallback_behavior("resign")
```

## Error Handling

The agent implements comprehensive error handling with graceful fallback mechanisms:

### API Failures
- **Connection Issues**: Retries up to `retry_attempts` times with exponential backoff
- **Timeout**: Falls back according to configured fallback behavior
- **Rate Limits**: Implements intelligent retry logic with configurable delays
- **Authentication Errors**: Provides clear error messages for API key issues

### Parsing Failures
- **Invalid UCI Format**: Falls back according to fallback behavior
- **Missing Tags**: Falls back according to fallback behavior
- **Illegal Moves**: Falls back according to fallback behavior
- **Empty Responses**: Falls back according to fallback behavior

### Template Validation
- **Missing Placeholders**: Gracefully handles missing variables with helpful warnings
- **Syntax Errors**: Validates basic template structure (balanced braces, non-empty)
- **Fallback Templates**: Automatically uses minimal working template if needed

### Example Error Handling
```python
# The agent handles errors automatically, but you can catch exceptions if needed
try:
    move, comment = agent.choose_move(board, legal_moves, [], "White")
    if move is None:
        print("Agent chose to resign")
    else:
        print(f"Agent chose: {move.uci()}")
except Exception as e:
    print(f"Unexpected error: {e}")
    # Agent automatically falls back according to fallback_behavior
```

### Fallback Behavior Configuration
```python
# Set fallback behavior during initialization
agent = OpenAIAgent(fallback_behavior="resign")

# Or update during runtime
agent.update_fallback_behavior("random_move")

# Available options:
# - "random_move": Choose a random legal move
# - "resign": Resign the game
```

## Template Customization and Management

The agent provides powerful template management capabilities for creating custom chess-playing personalities:

### Dynamic Template Updates
```python
# Update template during gameplay
midgame_template = """Quick tactical analysis needed!

Position: {FEN}
Your turn: {side_to_move}
Legal moves: {legal_moves_uci}

Think tactically and respond with <uci_move>move</uci_move>"""

agent.update_prompt_template(midgame_template)
```

### Template Validation
The agent now uses flexible validation that allows custom templates:
- **No Required Placeholders**: Create templates with only the variables you need
- **Basic Syntax Check**: Ensures balanced braces and non-empty content
- **Helpful Warnings**: Identifies potential issues without blocking usage
- **Graceful Fallback**: Automatically handles missing variables

### Template Examples by Use Case
```python
# Opening phase template
opening_template = """You are in the opening phase. Focus on development and control.

Board: {board_utf}
Legal moves: {legal_moves_uci}
Your turn: {side_to_move}

Develop pieces and control the center. Choose: <uci_move>move</uci_move>"""

# Endgame template
endgame_template = """Endgame position - precision is crucial.

Position: {FEN}
Legal moves: {legal_moves_uci}
History: {move_history_san}

Calculate carefully and choose: <uci_move>move</uci_move>"""

# Tactical template
tactical_template = """Tactical position - look for combinations!

Board: {board_utf}
Legal moves: {legal_moves_uci}
Last move: {last_move}

Find the best tactical continuation: <uci_move>move</uci_move>"""
```

### Template Management Tips
1. **Start Simple**: Begin with minimal templates and add complexity as needed
2. **Test Incrementally**: Test templates with simple positions first
3. **Use Variables Sparingly**: Only include the information the model needs
4. **Maintain Consistency**: Keep the move format requirements clear
5. **Monitor Performance**: Different templates may work better with different models

## Integration with Chess Environment

The OpenAI agent integrates seamlessly with the chess environment:

```python
from env import ChessEnvironment
from agents import OpenAIAgent, RandomAgent

# Create agents
openai_agent = OpenAIAgent(api_key="your-key")
random_agent = RandomAgent()

# Create environment
env = ChessEnvironment(openai_agent, random_agent, max_moves=30)

# Play a game
result = env.play_game(verbose=True)
print(f"Game result: {result['result']}")
```

## Testing

Run the OpenAI agent tests:

```bash
# Run all tests
python -m pytest tests/test_openai_agent.py -v

# Run specific test categories
python -m pytest tests/test_openai_agent.py::TestOpenAIAgent -v
python -m pytest tests/test_openai_agent.py::TestOpenAIAgentIntegration -v
```

### Test Categories
- **Unit Tests**: Mocked API calls for fast testing
- **Integration Tests**: Real API calls (requires API key)
- **Error Handling**: Tests fallback behavior and error recovery
- **Move Parsing**: Tests various response formats and edge cases

## Performance Considerations

### API Costs
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens (recommended for development)
- **GPT-5-mini**: ~$0.015 per 1K tokens (balanced performance and cost)
- **GPT-4**: ~$0.03 per 1K tokens (best chess understanding)

### Response Time
- **Typical**: 1-3 seconds per move
- **Network**: Depends on API latency and model size
- **Optimization**: Use smaller models for faster responses

### Rate Limits
- **Free Tier**: 3 requests per minute
- **Paid Tier**: Higher limits based on usage
- **Handling**: Built-in retry logic with exponential backoff

## Best Practices

### Prompt Design
1. **Be Specific**: Clearly specify the expected output format (UCI notation with tags)
2. **Include Context**: Provide only the information the model needs
3. **Set Constraints**: Limit response length and enforce UCI format
4. **Test Variations**: Experiment with different prompt styles and complexity levels
5. **Start Simple**: Begin with minimal templates and add complexity gradually
6. **Use Variables Sparingly**: Only include placeholders that add value

### Error Handling
1. **Graceful Degradation**: Always have fallback moves
2. **Logging**: Monitor API failures and parsing issues
3. **Retry Logic**: Implement appropriate retry strategies
4. **User Feedback**: Inform users when fallbacks are used

### Cost Optimization
1. **Model Selection**: Use appropriate model for your needs
2. **Token Limits**: Set reasonable max_tokens limits
3. **Caching**: Consider caching common positions
4. **Batch Processing**: Process multiple moves when possible

## Key Improvements in Updated Implementation

### Enhanced Template Flexibility
- **No Required Placeholders**: Create templates with only the variables you need
- **Custom Styles**: Support for minimal, focused, and comprehensive templates
- **Dynamic Switching**: Change templates during gameplay for different phases
- **Graceful Fallback**: Automatic handling of missing template variables

### Improved Error Handling
- **Configurable Fallback**: Choose between random moves and resignation
- **Better Validation**: Flexible template validation without strict requirements
- **Helpful Warnings**: Clear guidance for template issues
- **Automatic Recovery**: Seamless fallback to working templates

### Better Move Parsing
- **Strict UCI Format**: Consistent and reliable move extraction
- **Tagged Responses**: Required `<uci_move></uci_move>` format for clarity
- **No SAN Confusion**: Eliminates ambiguity between notation systems
- **Resignation Support**: Explicit handling of resignation decisions

## Limitations

### API Dependencies
- **Internet Required**: Cannot function without API access
- **Cost**: Each move incurs API call costs
- **Rate Limits**: Subject to OpenAI's rate limiting
- **Latency**: Network delays affect response time

### Model Behavior
- **Inconsistency**: Same position may produce different moves
- **Hallucination**: May suggest moves not in legal moves list
- **Context Limits**: Limited by model's context window
- **Training Data**: Quality depends on model's chess knowledge

### Evaluation Restrictions
**Note**: According to the SPEC, external API calls are not allowed during evaluation. This agent is intended for:
- Development and testing
- Prototyping chess strategies
- Learning prompt engineering techniques
- Fine-tuning preparation

For competition submissions, participants should fine-tune their own models based on the patterns learned from this agent.

## Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Error: OpenAI API key not provided
export OPENAI_API_KEY='your-actual-key-here'
```

#### Connection Failures
```python
# Test connection
if agent.test_connection():
    print("API connection successful")
else:
    print("Check your API key and internet connection")
```

#### Move Parsing Issues
```python
# Check prompt template
print(agent.get_prompt_template())

# Update template if needed
agent.update_prompt_template("Simpler template with {FEN} and {legal_moves_uci}")

# Test with minimal template
minimal_template = "Choose from: {legal_moves_uci}. Respond: <uci_move>move</uci_move>"
agent.update_prompt_template(minimal_template)
```

#### Template Issues
```python
# Check for missing placeholders
try:
    agent.update_prompt_template("Template with {nonexistent_variable}")
except Exception as e:
    print(f"Template error: {e}")

# Use only available placeholders
valid_template = "Position: {FEN}\nMoves: {legal_moves_uci}\nChoose: <uci_move>move</uci_move>"
agent.update_prompt_template(valid_template)
```

#### Example Game Issues
```bash
# If you get "Stockfish binary not found" error:
# macOS: brew install stockfish
# Ubuntu: sudo apt install stockfish
# Windows: Download from https://stockfishchess.org/download/

# If OpenAI API calls fail:
export OPENAI_API_KEY='your-actual-api-key'
python run_game.py

# If you want to see more detailed output:
python run_game.py | tee game_log.txt

# To analyze the generated PGN file:
python -c "
import chess.pgn
with open('game.pgn') as f:
    game = chess.pgn.read_game(f)
    print(f'Game result: {game.headers[\"Result\"]}')
    print(f'Termination: {game.headers.get(\"Termination\", \"Unknown\")}')
"
```

#### Performance Issues
```python
# Reduce model size
agent = OpenAIAgent(model="gpt-3.5-turbo")

# Reduce token limit
agent.update_generation_params(max_tokens=20)

# Increase timeout for complex positions
agent = OpenAIAgent(timeout=60.0)
```

## Examples

### Complete Gameplay Example: OpenAI vs Stockfish

The `run_game.py` script demonstrates a complete chess game between an OpenAI agent and a Stockfish Level 1 agent, showcasing template customization and enhanced gameplay features.

#### Running the Example
```bash
# Make sure you have your OpenAI API key set
export OPENAI_API_KEY='your-api-key-here'

# Run the example game
python run_game.py
```

#### What the Example Demonstrates
- **Template Customization**: Shows how to create and update custom prompt templates
- **Random Color Assignment**: Agents randomly assigned White/Black for fairness
- **Dynamic Template Switching**: Changes templates during gameplay
- **Enhanced Game Display**: Shows board positions, move analysis, and game progress
- **PGN Export**: Saves complete game to `game.pgn` file
- **Game Termination Detection**: Proper handling of checkmate, stalemate, and draws

#### Key Features Demonstrated
```python
# Custom template creation
custom_template = """You are a tactical chess player. Analyze the position and choose the best move.

BOARD POSITION:
{board_utf}

GAME STATE:
- Your turn: {side_to_move}
- Available moves: {legal_moves_uci}
- Game history: {move_history_san}

INSTRUCTIONS:
1. Evaluate the position for tactical opportunities
2. Choose the strongest move from the legal options above
3. Respond with your move in UCI notation wrapped in <uci_move></uci_move> tags

EXAMPLE: <uci_move>e2e4</uci_move>"""

# Dynamic template switching
midgame_template = """Quick tactical analysis needed!

Position: {FEN}
Your turn: {side_to_move}
Legal moves: {legal_moves_uci}

Think tactically and respond with <uci_move>move</uci_move>"""

# Apply templates
openai_agent.update_prompt_template(custom_template)
openai_agent.update_prompt_template(midgame_template)
```

#### Sample Output
```
=== OpenAI vs Stockfish Level 1 Game ===

🎨 Customizing OpenAI agent prompt template...

📋 Available template variables:
  • {board_utf}     - Visual board with Unicode pieces
  • {FEN}           - FEN notation of current position
  • {side_to_move}  - Which side is to move ('White' or 'Black')
  • {legal_moves_uci} - Available moves in UCI notation (e.g., 'e2e4')
  • {legal_moves_san} - Available moves in SAN notation (e.g., 'e4')
  • {move_history_uci} - Game history in UCI notation
  • {move_history_san} - Game history in SAN notation
  • {last_move}     - Description of the last move played

🎲 Randomly assigning colors...
🏆 White: OpenAIAgent
🏆 Black: StockfishAgent

🎮 Starting the game with custom template...
============================================================
Starting new game: OpenAIAgent (White) vs StockfishAgent (Black)

Move 1: White's turn
Legal moves: g1h3, g1f3, b1c3, b1a3, h2h3, ... and 15 more
✅ White plays: e2e4
   Comment: <uci_move>e2e4</uci_move>

Position after 1. e2e4:
  8    ♜    ♞    ♝    ♛    ♚    ♝    ♞    ♜  
  7    ♟    ♟    ♟    ♟    ♟    ♟    ♟    ♟  
  6    ·    ·    ·    ·    ·    ·    ·    ·  
  5    ·    ·    ·    ·    ·    ·    ·    ·  
  4    ·    ·    ·    ·    ♙    ·    ·    ·  
  3    ·    ·    ·    ·    ·    ·    ·    ·  
  2    ♙    ♙    ♙    ♙    ·    ♙    ♙    ♙  
  1    ♖    ♘    ♗    ♕    ♔    ♗    ♘    ♖  
       a    b    c    d    e    f    g    h  
```

#### What You'll See
1. **Template Customization**: Watch as the agent's prompt template is customized
2. **Random Color Assignment**: See which agent gets White vs Black
3. **Game Progress**: Follow each move with board visualization
4. **Move Analysis**: Read the AI's reasoning for each move
5. **Game Results**: See the final outcome and termination reason
6. **PGN Export**: Game saved to `game.pgn` for analysis

#### Expected Game Duration
- **Typical Game**: 20-50 moves (depending on agent performance)
- **Time per Move**: 1-3 seconds for OpenAI, <1 second for Stockfish
- **Total Runtime**: 2-5 minutes for a complete game

#### Output Files
- **`game.pgn`**: Complete game in PGN format with metadata
- **Console Output**: Real-time game progress and analysis
- **Final Summary**: Game result, move count, and termination reason

### Basic Game Loop
```python
import chess
from agents import OpenAIAgent

def play_game_with_openai():
    board = chess.Board()
    agent = OpenAIAgent(api_key="your-key")
    
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        # Get move from OpenAI
        move, comment = agent.choose_move(board, legal_moves, [], 
                                        "White" if board.turn else "Black")
        
        if move is None:
            print("Agent resigned")
            break
            
        # Make the move
        board.push(move)
        print(f"Move: {board.san(move)}")
        print(f"Comment: {comment}")
    
    print(f"Game result: {board.outcome()}")
```

### Custom Prompt Engineering
```python
# Create a specialized prompt for endgames
endgame_prompt = """You are a chess endgame expert. Analyze this position carefully.

Position: {FEN}
Legal moves: {legal_moves_uci}
Your side: {side_to_move}

Focus on:
1. King safety
2. Pawn advancement
3. Piece coordination

Respond with your move in UCI notation wrapped in <uci_move></uci_move> tags."""

agent.update_prompt_template(endgame_prompt)

# Create a tactical opening template
tactical_opening = """You are a tactical player in the opening phase.

Board: {board_utf}
Legal moves: {legal_moves_uci}
Game history: {move_history_san}

Look for tactical opportunities and respond with <uci_move>move</uci_move>"""

agent.update_prompt_template(tactical_opening)
```

### Parameter Tuning
```python
# Create different agents for different playing styles
aggressive_agent = OpenAIAgent(
    model="gpt-5-mini",
    temperature=0.8,  # More creative/aggressive
    max_tokens=100
)

conservative_agent = OpenAIAgent(
    model="gpt-5-mini", 
    temperature=0.0,  # More deterministic/defensive
    max_tokens=50
)
```

### Customizing the Example Game
You can modify `run_game.py` to experiment with different scenarios:

```python
# Change the Stockfish skill level for different difficulty
stockfish_agent = StockfishAgent(
    skill_level=10,        # Level 10 (moderate strength)
    depth=10,              # Higher depth for better play
    time_limit_ms=2000     # 2 seconds per move
)

# Use different OpenAI models
openai_agent = OpenAIAgent(
    model="gpt-3.5-turbo",  # Faster, cheaper
    max_completion_tokens=300
)

# Create custom templates for specific game phases
opening_template = """You are in the opening phase. Focus on development.

Board: {board_utf}
Legal moves: {legal_moves_uci}

Develop pieces and control the center: <uci_move>move</uci_move>"""

midgame_template = """Midgame position - look for tactics!

Position: {FEN}
Legal moves: {legal_moves_uci}

Find the best tactical move: <uci_move>move</uci_move>"""

# Apply different templates at different phases
if move_count < 10:
    openai_agent.update_prompt_template(opening_template)
else:
    openai_agent.update_prompt_template(midgame_template)
```

### Running Multiple Games
```python
# Modify run_game.py to run multiple games
for game_num in range(3):
    print(f"\n=== Game {game_num + 1} ===")
    result = env.play_game(verbose=True)
    print(f"Game {game_num + 1} result: {result['result']}")
    
    # Save each game with unique filename
    pgn_content = env._generate_pgn_content(include_metadata=True)
    with open(f"game_{game_num + 1}.pgn", "w") as f:
        f.write(pgn_content)
```

## Contributing

To contribute to the OpenAI agent:

1. **Follow Testing**: Add tests for new features
2. **Document Changes**: Update this documentation
3. **Error Handling**: Ensure robust error handling
4. **Performance**: Consider API costs and response times
5. **Compatibility**: Maintain compatibility with the chess environment

## License

This agent is part of the AIcrowd Chess Challenge and follows the same licensing terms as the main project.
