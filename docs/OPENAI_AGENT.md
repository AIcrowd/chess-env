# OpenAI Chess Agent

The `OpenAIAgent` is a chess-playing agent that uses OpenAI's API to make chess moves. It follows the SPEC requirements for prompt templates, move parsing, and integration with the chess environment.

## Features

- **OpenAI API Integration**: Uses OpenAI's GPT models for chess move selection
- **Configurable Prompt Templates**: Supports customizable prompts with placeholders
- **Robust Move Parsing**: Intelligently parses model responses to extract valid moves
- **Fallback Handling**: Gracefully handles API failures and parsing errors
- **Parameter Configuration**: Configurable generation parameters (temperature, max_tokens, etc.)
- **Error Recovery**: Falls back to legal moves when API calls fail

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

The agent uses configurable prompt templates with placeholders that get filled with actual game data:

### Default Template
```python
DEFAULT_PROMPT_TEMPLATE = """You are a chess grandmaster. Analyze the board and provide the best move.

Current board (FEN): {FEN}
Last move: {last_move}
Legal moves available: {legal_moves}
It is your turn as {side_to_move}.

What move should you play? Please output your chosen move in Standard Algebraic Notation (SAN) format only, such as "e4", "Nf3", "O-O", etc. Do not include any explanation or additional text."""
```

### Custom Templates
```python
custom_template = """You are a chess expert. Choose the best move from the available options.

Board position: {FEN}
Available moves: {legal_moves}
Your turn as: {side_to_move}

Respond with only the move in SAN notation (e.g., e4, Nf3)."""

agent.update_prompt_template(custom_template)
```

### Available Placeholders
- `{FEN}`: Current board position in Forsyth-Edwards Notation
- `{legal_moves}`: List of available legal moves
- `{side_to_move}`: Which side is to move ("White" or "Black")
- `{last_move}`: Description of the last move played
- `{move_history}`: Sequence of moves played so far

## Move Parsing

The agent intelligently parses model responses to extract valid chess moves:

### Supported Formats
- **Standard Algebraic Notation (SAN)**: `e4`, `Nf3`, `O-O`, `exd5`
- **Case Insensitive**: `E4`, `nf3`, `o-o` all work
- **With Context**: `"I would play e4"` → extracts `e4`
- **Complex Responses**: `"The best move is Nf3 followed by..."` → extracts `Nf3`

### Fallback Behavior
If the agent cannot parse a valid move from the model's response, it falls back to the first legal move available. This ensures the game can continue even if the model produces unexpected output.

## Error Handling

### API Failures
- **Connection Issues**: Retries up to `retry_attempts` times
- **Timeout**: Falls back to first legal move after timeout
- **Rate Limits**: Implements exponential backoff

### Parsing Failures
- **Invalid Moves**: Falls back to first legal move
- **Empty Responses**: Falls back to first legal move
- **Malformed Output**: Attempts pattern matching before falling back

### Example Error Handling
```python
try:
    move = agent.choose_move(board, legal_moves, [], "White")
except Exception as e:
    print(f"Agent failed: {e}")
    # Agent automatically falls back to first legal move
```

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
1. **Be Specific**: Clearly specify the expected output format
2. **Include Context**: Provide board state, legal moves, and game history
3. **Set Constraints**: Limit response length and format
4. **Test Variations**: Experiment with different prompt styles

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
agent.update_prompt_template("Simpler template with {FEN} and {legal_moves}")
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
        move = agent.choose_move(board, legal_moves, [], 
                               "White" if board.turn else "Black")
        
        # Make the move
        board.push(move)
        print(f"Move: {board.san(move)}")
    
    print(f"Game result: {board.outcome()}")
```

### Custom Prompt Engineering
```python
# Create a specialized prompt for endgames
endgame_prompt = """You are a chess endgame expert. Analyze this position carefully.

Position: {FEN}
Legal moves: {legal_moves}
Your side: {side_to_move}

Focus on:
1. King safety
2. Pawn advancement
3. Piece coordination

Respond with only the move in SAN notation."""

agent.update_prompt_template(endgame_prompt)
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

## Contributing

To contribute to the OpenAI agent:

1. **Follow Testing**: Add tests for new features
2. **Document Changes**: Update this documentation
3. **Error Handling**: Ensure robust error handling
4. **Performance**: Consider API costs and response times
5. **Compatibility**: Maintain compatibility with the chess environment

## License

This agent is part of the AIcrowd Chess Challenge and follows the same licensing terms as the main project.
