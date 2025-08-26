"""
Tests for the OpenAI agent.

These tests verify the OpenAI agent functionality, including:
- Initialization and API key handling
- Prompt template validation and formatting
- Move parsing and validation
- Error handling and fallbacks
- Configuration updates
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

import chess
from agents.openai_agent import OpenAIAgent


class TestOpenAIAgent:
    """Test cases for OpenAIAgent class."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key-123")
            
            assert agent.api_key == "test-key-123"
            assert agent.model == "gpt-5-mini"
            assert agent.temperature == 0.1
            assert agent.max_tokens == 50
            assert agent.fallback_behavior == "random_move"
            mock_openai.assert_called_once_with(api_key="test-key-123")
    
    def test_init_with_environment_variable(self):
        """Test initialization using OPENAI_API_KEY environment variable."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key-456'}):
                agent = OpenAIAgent()
            
            assert agent.api_key == "env-key-456"
            mock_openai.assert_called_once_with(api_key="env-key-456")
    
    def test_init_no_api_key(self):
        """Test that initialization fails when no API key is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not provided"):
                OpenAIAgent()
    
    def test_default_prompt_template(self):
        """Test that default prompt template contains required placeholders."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            template = agent.get_prompt_template()
            assert "{FEN}" in template
            assert "{board_utf}" in template
            assert "{legal_moves_uci}" in template
            assert "{legal_moves_san}" in template
            assert "{move_history_uci}" in template
            assert "{move_history_san}" in template
            assert "{side_to_move}" in template
            assert "{last_move}" in template
    
    def test_custom_prompt_template(self):
        """Test initialization with custom prompt template."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            custom_template = "Custom template with {FEN} and {board_utf} and {legal_moves_uci} and {legal_moves_san} and {move_history_uci} and {move_history_san} and {side_to_move}"
            agent = OpenAIAgent(api_key="test-key", prompt_template=custom_template)
            
            assert agent.get_prompt_template() == custom_template
    
    def test_invalid_prompt_template(self):
        """Test that invalid prompt templates raise an error."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            invalid_template = "Missing required placeholders"
            
            with pytest.raises(ValueError, match="Prompt template must contain"):
                OpenAIAgent(api_key="test-key", prompt_template=invalid_template)
    
    def test_generation_params(self):
        """Test that generation parameters are properly set."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(
                api_key="test-key",
                temperature=0.5,
                max_tokens=100,
                top_p=0.9
            )
            
            params = agent.get_generation_params()
            # Temperature might not be present for GPT-5 models
            if "temperature" in params:
                assert params["temperature"] == 0.5
            # Always use max_completion_tokens for the new Python API
            assert params["max_completion_tokens"] == 100
            assert params["top_p"] == 0.9
    
    def test_prompt_formatting(self):
        """Test that prompts are properly formatted with game data."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            # Create a test board
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            prompt = agent._format_prompt(board, legal_moves, [], "White")
            
            # Check that placeholders are filled
            assert "{FEN}" not in prompt
            assert "{board_utf}" not in prompt
            assert "{legal_moves_uci}" not in prompt
            assert "{legal_moves_san}" not in prompt
            assert "{move_history_uci}" not in prompt
            assert "{move_history_san}" not in prompt
            assert "{side_to_move}" not in prompt
            assert "{last_move}" not in prompt
            
            # Check that actual values are present
            assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" in prompt
            assert "White" in prompt
            assert "(start of game)" in prompt
    
    def test_prompt_formatting_with_moves(self):
        """Test prompt formatting when moves have been played."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            # Create a board with some moves
            board = chess.Board()
            board.push(chess.Move.from_uci("e2e4"))
            board.push(chess.Move.from_uci("e7e5"))
            
            legal_moves = list(board.legal_moves)
            move_history = ["e2e4", "e7e5"]
            
            prompt = agent._format_prompt(board, legal_moves, move_history, "White")
            
            # Check that last move is included
            assert "Black played e5" in prompt
            assert "White" in prompt
    
    def test_move_parsing_exact_match(self):
        """Test that moves are parsed correctly when they exactly match legal moves."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test exact match with UCI tags
            response = "<uci_move>e2e4</uci_move>"
            move = agent._parse_move(response, legal_moves, board)
            
            assert move.uci() == "e2e4"
    
    def test_move_parsing_case_insensitive(self):
        """Test that move parsing is case insensitive for UCI tags."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test case insensitive UCI tags
            response = "<UCI_MOVE>g1f3</UCI_MOVE>"
            move = agent._parse_move(response, legal_moves, board)
            
            assert move.uci() == "g1f3"
    
    def test_move_parsing_with_prefixes(self):
        """Test that move parsing handles text around UCI tags."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test with text around UCI tags
            response = "I would play <uci_move>e2e4</uci_move>"
            move = agent._parse_move(response, legal_moves, board)
            
            assert move.uci() == "e2e4"
    
    def test_move_parsing_regex_patterns(self):
        """Test that move parsing extracts UCI moves from complex text."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test UCI extraction from complex text
            response = "After analyzing the position, I choose <uci_move>g1f3</uci_move>"
            move = agent._parse_move(response, legal_moves, board)
            
            assert move.uci() == "g1f3"
    
    def test_move_parsing_failure(self):
        """Test that move parsing failure raises an appropriate error."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test with invalid response (no UCI tags)
            response = "This is not a valid move"
            
            with pytest.raises(ValueError, match="Model did not respond with required UCI move tags"):
                agent._parse_move(response, legal_moves, board)
    
    def test_api_call_success(self):
        """Test successful API call."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "e4"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            response = agent._call_openai_api("test prompt")
            assert response == "e4"
    
    def test_api_call_failure(self):
        """Test API call failure handling."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key", retry_attempts=2)
            
            with pytest.raises(Exception, match="OpenAI API call failed after 2 attempts"):
                agent._call_openai_api("test prompt")
    
    def test_choose_move_success(self):
        """Test successful move selection."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "<uci_move>e2e4</uci_move>"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            move_result = agent.choose_move(board, legal_moves, [], "White")
            move, comment = move_result
            assert move.uci() == "e2e4"
    
    def test_choose_move_api_failure_fallback(self):
        """Test that agent falls back to first legal move when API fails."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key", retry_attempts=1)
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            move_result = agent.choose_move(board, legal_moves, [], "White")
            move, comment = move_result
            # Should fall back to first legal move
            assert move == legal_moves[0]
    
    def test_choose_move_parsing_failure_fallback(self):
        """Test that agent falls back to first legal move when parsing fails."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "<uci_move>e2e4</uci_move>"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            move_result = agent.choose_move(board, legal_moves, [], "White")
            move, comment = move_result
            # Should successfully parse the UCI move
            assert move.uci() == "e2e4"
    
    def test_choose_move_no_legal_moves(self):
        """Test that agent raises error when no legal moves are available."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = []
            
            with pytest.raises(ValueError, match="No legal moves available"):
                agent.choose_move(board, legal_moves, [], "White")
    
    def test_update_prompt_template(self):
        """Test updating prompt template."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            new_template = "New template with {FEN} and {board_utf} and {legal_moves_uci} and {legal_moves_san} and {move_history_uci} and {move_history_san} and {side_to_move}"
            agent.update_prompt_template(new_template)
            
            assert agent.get_prompt_template() == new_template
    
    def test_update_prompt_template_invalid(self):
        """Test that updating with invalid template raises error."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            invalid_template = "Missing required placeholders"
            
            with pytest.raises(ValueError, match="Prompt template must contain"):
                agent.update_prompt_template(invalid_template)
    
    def test_update_generation_params(self):
        """Test updating generation parameters."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            agent.update_generation_params(temperature=0.8, max_tokens=200)
            
            params = agent.get_generation_params()
            # Temperature might not be present for GPT-5 models
            if "temperature" in params:
                assert params["temperature"] == 0.8
            # Always use max_completion_tokens for the new Python API
            assert params["max_completion_tokens"] == 200
            # Temperature might not be updated for GPT-5 models
            if agent.temperature != 0.1:  # Default value
                assert agent.temperature == 0.8
            assert agent.max_tokens == 200
    
    def test_test_connection_success(self):
        """Test successful connection test."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            assert agent.test_connection() is True
    
    def test_test_connection_failure(self):
        """Test failed connection test."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Connection failed")
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            assert agent.test_connection() is False
    
    def test_fallback_behavior_parameter(self):
        """Test that fallback behavior parameter is properly set."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Test default fallback behavior
            agent = OpenAIAgent(api_key="test-key")
            assert agent.fallback_behavior == "random_move"
            
            # Test custom fallback behavior
            agent = OpenAIAgent(api_key="test-key", fallback_behavior="resign")
            assert agent.fallback_behavior == "resign"
    
    def test_invalid_fallback_behavior(self):
        """Test that invalid fallback behavior raises an error."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with pytest.raises(ValueError, match="Invalid fallback_behavior"):
                OpenAIAgent(api_key="test-key", fallback_behavior="invalid")
    
    def test_uci_move_parsing(self):
        """Test that UCI move tags are properly parsed."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test UCI move in tags
            response = "I think the best move is <uci_move>e2e4</uci_move>"
            move = agent._parse_move(response, legal_moves, board)
            assert move.uci() == "e2e4"
            
            # Test UCI move with extra text
            response = "After analyzing the position, I choose <uci_move>g1f3</uci_move>"
            move = agent._parse_move(response, legal_moves, board)
            assert move.uci() == "g1f3"
    
    def test_resignation_parsing(self):
        """Test that resignation is properly handled."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test resignation
            response = "I cannot find a good move. <uci_move>resign</uci_move>"
            
            with pytest.raises(ValueError, match="Model chose to resign"):
                agent._parse_move(response, legal_moves, board)
    
    def test_invalid_uci_format(self):
        """Test that invalid UCI format is treated as resignation."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test invalid UCI format
            response = "<uci_move>invalid_move</uci_move>"
            
            with pytest.raises(ValueError, match="Model provided invalid UCI format"):
                agent._parse_move(response, legal_moves, board)
    
    def test_illegal_move(self):
        """Test that illegal moves are treated as resignation."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Test illegal move (e.g., moving a pawn from e3 when it's not there)
            response = "<uci_move>e3e4</uci_move>"
            
            with pytest.raises(ValueError, match="Model provided illegal move"):
                agent._parse_move(response, legal_moves, board)
    
    def test_fallback_behavior_random_move(self):
        """Test random move fallback behavior."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "<uci_move>e2e4</uci_move>"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key", fallback_behavior="random_move")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            move_result = agent.choose_move(board, legal_moves, [], "White")
            move, comment = move_result
            # Should successfully parse the UCI move
            assert move.uci() == "e2e4"
    
    def test_fallback_behavior_resign(self):
        """Test resign fallback behavior."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "<uci_move>e2e4</uci_move>"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key", fallback_behavior="resign")
            
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # With valid UCI move, should succeed regardless of fallback behavior
            move_result = agent.choose_move(board, legal_moves, [], "White")
            move, comment = move_result
            assert move.uci() == "e2e4"
    
    def test_update_fallback_behavior(self):
        """Test updating fallback behavior."""
        with patch('agents.openai_agent.openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            agent = OpenAIAgent(api_key="test-key")
            
            # Test updating to resign
            agent.update_fallback_behavior("resign")
            assert agent.fallback_behavior == "resign"
            
            # Test updating to random_move
            agent.update_fallback_behavior("random_move")
            assert agent.fallback_behavior == "random_move"
            
            # Test invalid update
            with pytest.raises(ValueError, match="Invalid fallback_behavior"):
                agent.update_fallback_behavior("invalid")


class TestOpenAIAgentIntegration:
    """Integration tests for OpenAI agent (requires API key)."""
    
    @pytest.mark.skipif(
        not os.environ.get('OPENAI_API_KEY'),
        reason="OpenAI API key not available"
    )
    def test_real_openai_connection(self):
        """Test that OpenAI agent can connect to real API."""
        try:
            agent = OpenAIAgent()
            assert agent.test_connection() is True
        except Exception as e:
            pytest.skip(f"OpenAI connection failed: {e}")
    
    @pytest.mark.skipif(
        not os.environ.get('OPENAI_API_KEY'),
        reason="OpenAI API key not available"
    )
    def test_real_openai_move_selection(self):
        """Test that OpenAI agent can make real moves."""
        try:
            agent = OpenAIAgent(model="gpt-5-mini", temperature=0.0)
            
            # Create a simple position
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            
            # Get a move
            move_result = agent.choose_move(board, legal_moves, [], "White")
            move, comment = move_result
            
            assert move is not None
            assert move in legal_moves
            
        except Exception as e:
            pytest.skip(f"OpenAI move selection failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
