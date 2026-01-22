"""Tests for AIGenerator class"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorGenerateResponse(unittest.TestCase):
    """Tests for AIGenerator.generate_response() method"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "test-model"

    @patch('ai_generator.anthropic.Anthropic')
    def test_passes_tools_to_api(self, mock_anthropic_class):
        """Tools are included in API call when provided"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response text")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(self.api_key, self.model)
        tools = [{"name": "test_tool", "description": "Test"}]

        # Act
        generator.generate_response(query="test", tools=tools)

        # Assert
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["tools"], tools)
        self.assertEqual(call_kwargs["tool_choice"], {"type": "auto"})

    @patch('ai_generator.anthropic.Anthropic')
    def test_includes_system_prompt(self, mock_anthropic_class):
        """SYSTEM_PROMPT is included in system parameter"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(self.api_key, self.model)

        # Act
        generator.generate_response(query="test")

        # Assert
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertIn("system", call_kwargs)
        self.assertIn("AI assistant", call_kwargs["system"])

    @patch('ai_generator.anthropic.Anthropic')
    def test_includes_conversation_history(self, mock_anthropic_class):
        """Conversation history is appended to system prompt"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(self.api_key, self.model)
        history = "User: Previous question\nAssistant: Previous answer"

        # Act
        generator.generate_response(query="test", conversation_history=history)

        # Assert
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertIn(history, call_kwargs["system"])
        self.assertIn("Previous conversation:", call_kwargs["system"])

    @patch('ai_generator.anthropic.Anthropic')
    def test_returns_text_for_direct_response(self, mock_anthropic_class):
        """Returns text when no tool use is needed"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(self.api_key, self.model)

        # Act
        result = generator.generate_response(query="What is 2+2?")

        # Assert
        self.assertEqual(result, "Direct answer")

    @patch('ai_generator.anthropic.Anthropic')
    def test_triggers_tool_execution_on_tool_use(self, mock_anthropic_class):
        """When stop_reason='tool_use', tool execution is triggered"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "python"}

        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [tool_use_block]

        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Final answer after tool")]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        generator = AIGenerator(self.api_key, self.model)
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="search for python",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python"
        )
        self.assertEqual(result, "Final answer after tool")


class TestAIGeneratorHandleToolExecution(unittest.TestCase):
    """Tests for AIGenerator._handle_tool_execution() method"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_extracts_tool_params(self, mock_anthropic_class):
        """Correct tool name and input are extracted from response"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "get_course_outline"
        tool_use_block.id = "tool_456"
        tool_use_block.input = {"course_name": "MCP Course"}

        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [tool_use_block]

        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Outline info")]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course outline data"

        # Act
        generator.generate_response(
            query="show me MCP course outline",
            tools=[{"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )

        # Assert
        mock_tool_manager.execute_tool.assert_called_with(
            "get_course_outline",
            course_name="MCP Course"
        )

    @patch('ai_generator.anthropic.Anthropic')
    def test_passes_result_to_follow_up(self, mock_anthropic_class):
        """Tool result is included in follow-up API call messages"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_789"
        tool_use_block.input = {"query": "test"}

        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [tool_use_block]

        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Final")]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        # Act
        generator.generate_response(
            query="test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Check the second API call
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Find the tool result message
        tool_result_msg = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "tool_result":
                        tool_result_msg = item
                        break

        self.assertIsNotNone(tool_result_msg)
        self.assertEqual(tool_result_msg["content"], "Search results here")
        self.assertEqual(tool_result_msg["tool_use_id"], "tool_789")

    @patch('ai_generator.anthropic.Anthropic')
    def test_builds_correct_message_structure(self, mock_anthropic_class):
        """Follow-up call has proper message sequence"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_abc"
        tool_use_block.input = {"query": "test"}

        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [tool_use_block]

        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Done")]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        # Act
        generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Check message structure
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Should have: user query, assistant tool use, user tool result
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")

    @patch('ai_generator.anthropic.Anthropic')
    def test_returns_final_response(self, mock_anthropic_class):
        """Returns text from the second API call"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_def"
        tool_use_block.input = {"query": "test"}

        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [tool_use_block]

        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="This is the final answer")]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        # Act
        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "This is the final answer")


class TestAIGeneratorWithNoToolManager(unittest.TestCase):
    """Tests for edge cases when tool_manager is not provided"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_use_without_manager_returns_text(self, mock_anthropic_class):
        """When tool_use but no tool_manager, should still return content text"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        text_block = Mock()
        text_block.text = "I'll search for that"

        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [text_block]  # Has text even with tool_use

        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator("key", "model")

        # Act - No tool_manager provided
        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=None  # No tool manager
        )

        # Assert - Returns text content instead of executing tool
        self.assertEqual(result, "I'll search for that")


class TestAIGeneratorSequentialToolCalling(unittest.TestCase):
    """Tests for sequential tool calling with up to 2 rounds"""

    def _create_tool_use_block(self, name, tool_id, input_dict):
        """Helper to create a mock tool use block"""
        block = Mock()
        block.type = "tool_use"
        block.name = name
        block.id = tool_id
        block.input = input_dict
        return block

    def _create_tool_response(self, tool_blocks):
        """Helper to create a mock tool_use response"""
        response = Mock()
        response.stop_reason = "tool_use"
        response.content = tool_blocks
        return response

    def _create_text_response(self, text):
        """Helper to create a mock text response"""
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text=text)]
        return response

    @patch('ai_generator.anthropic.Anthropic')
    def test_two_sequential_tool_calls(self, mock_anthropic_class):
        """Supports two sequential tool calls in separate API rounds"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: get_course_outline
        tool_block_1 = self._create_tool_use_block(
            "get_course_outline", "tool_1", {"course_name": "MCP Course"}
        )
        response_1 = self._create_tool_response([tool_block_1])

        # Second response: search_course_content
        tool_block_2 = self._create_tool_use_block(
            "search_course_content", "tool_2", {"query": "MCP basics"}
        )
        response_2 = self._create_tool_response([tool_block_2])

        # Third response: final answer
        response_3 = self._create_text_response("Complete answer after 2 tools")

        mock_client.messages.create.side_effect = [response_1, response_2, response_3]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Course outline", "Search results"]
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="Find courses similar to MCP lesson 4",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(result, "Complete answer after 2 tools")

    @patch('ai_generator.anthropic.Anthropic')
    def test_max_rounds_enforced(self, mock_anthropic_class):
        """Terminates after MAX_TOOL_ROUNDS even if Claude wants more tools"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create tool blocks for 3 rounds (but only 2 should execute)
        tool_block_1 = self._create_tool_use_block("tool_a", "id_1", {"q": "1"})
        tool_block_2 = self._create_tool_use_block("tool_b", "id_2", {"q": "2"})
        tool_block_3 = self._create_tool_use_block("tool_c", "id_3", {"q": "3"})

        response_1 = self._create_tool_response([tool_block_1])
        response_2 = self._create_tool_response([tool_block_2])
        # Third response still has tool_use but should be ignored
        response_3 = self._create_tool_response([tool_block_3])

        mock_client.messages.create.side_effect = [response_1, response_2, response_3]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"
        tools = [{"name": "tool_a"}, {"name": "tool_b"}, {"name": "tool_c"}]

        # Act
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Only 2 tool executions (MAX_TOOL_ROUNDS = 2)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        # 3 API calls: initial + round 1 + round 2
        self.assertEqual(mock_client.messages.create.call_count, 3)

    @patch('ai_generator.anthropic.Anthropic')
    def test_tools_included_in_second_api_call(self, mock_anthropic_class):
        """Tools parameter is included in second API call for potential follow-up"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_block = self._create_tool_use_block("search", "id_1", {"q": "test"})
        response_1 = self._create_tool_response([tool_block])
        response_2 = self._create_text_response("Answer")

        mock_client.messages.create.side_effect = [response_1, response_2]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"
        tools = [{"name": "search"}]

        # Act
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Second call should have tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        self.assertIn("tools", second_call_kwargs)
        self.assertEqual(second_call_kwargs["tools"], tools)

    @patch('ai_generator.anthropic.Anthropic')
    def test_tools_excluded_after_max_rounds(self, mock_anthropic_class):
        """Tools parameter is excluded in final call after max rounds reached"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_block_1 = self._create_tool_use_block("tool_a", "id_1", {"q": "1"})
        tool_block_2 = self._create_tool_use_block("tool_b", "id_2", {"q": "2"})

        response_1 = self._create_tool_response([tool_block_1])
        response_2 = self._create_tool_response([tool_block_2])
        response_3 = self._create_text_response("Final answer")

        mock_client.messages.create.side_effect = [response_1, response_2, response_3]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"
        tools = [{"name": "tool_a"}, {"name": "tool_b"}]

        # Act
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Third call (after max rounds) should NOT have tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", third_call_kwargs)

    @patch('ai_generator.anthropic.Anthropic')
    def test_early_termination_on_text_response(self, mock_anthropic_class):
        """Terminates early when Claude returns text without tool_use"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_block = self._create_tool_use_block("search", "id_1", {"q": "test"})
        response_1 = self._create_tool_response([tool_block])
        response_2 = self._create_text_response("Got enough info, here's the answer")

        mock_client.messages.create.side_effect = [response_1, response_2]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        tools = [{"name": "search"}]

        # Act
        result = generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Only 2 API calls (early termination)
        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)
        self.assertEqual(result, "Got enough info, here's the answer")

    @patch('ai_generator.anthropic.Anthropic')
    def test_message_accumulation_across_rounds(self, mock_anthropic_class):
        """Messages accumulate correctly across multiple tool rounds"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_block_1 = self._create_tool_use_block("tool_a", "id_1", {"q": "1"})
        tool_block_2 = self._create_tool_use_block("tool_b", "id_2", {"q": "2"})

        response_1 = self._create_tool_response([tool_block_1])
        response_2 = self._create_tool_response([tool_block_2])
        response_3 = self._create_text_response("Final")

        mock_client.messages.create.side_effect = [response_1, response_2, response_3]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        tools = [{"name": "tool_a"}, {"name": "tool_b"}]

        # Act
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Third call should have accumulated messages from both rounds
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]

        # Should have: user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertEqual(messages[4]["role"], "user")

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_error_handled_gracefully(self, mock_anthropic_class):
        """Tool execution errors are captured and included in results"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_block = self._create_tool_use_block("failing_tool", "id_1", {"q": "test"})
        response_1 = self._create_tool_response([tool_block])
        response_2 = self._create_text_response("Handled the error gracefully")

        mock_client.messages.create.side_effect = [response_1, response_2]

        generator = AIGenerator("key", "model")
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed!")
        tools = [{"name": "failing_tool"}]

        # Act
        result = generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Should not raise, returns response
        self.assertEqual(result, "Handled the error gracefully")

        # Check error was included in tool result
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_msg = messages[2]["content"][0]
        self.assertIn("Error executing tool", tool_result_msg["content"])


if __name__ == "__main__":
    unittest.main()
