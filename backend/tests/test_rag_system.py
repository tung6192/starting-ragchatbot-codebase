"""Tests for RAGSystem class"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGSystemQuery(unittest.TestCase):
    """Tests for RAGSystem.query() method"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    def setUp(self, mock_outline_tool, mock_search_tool, mock_tool_manager_class,
              mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Set up test fixtures with mocked dependencies"""
        # Store mock classes for later use
        self.mock_doc_processor = mock_doc_processor
        self.mock_vector_store_class = mock_vector_store
        self.mock_ai_generator_class = mock_ai_generator
        self.mock_session_manager_class = mock_session_manager
        self.mock_tool_manager_class = mock_tool_manager_class
        self.mock_search_tool_class = mock_search_tool
        self.mock_outline_tool_class = mock_outline_tool

        # Create mock instances
        self.mock_vector_store = Mock()
        self.mock_ai_generator = Mock()
        self.mock_session_manager = Mock()
        self.mock_tool_manager = Mock()
        self.mock_search_tool = Mock()
        self.mock_outline_tool = Mock()

        # Configure mock classes to return our mock instances
        mock_vector_store.return_value = self.mock_vector_store
        mock_ai_generator.return_value = self.mock_ai_generator
        mock_session_manager.return_value = self.mock_session_manager
        mock_tool_manager_class.return_value = self.mock_tool_manager
        mock_search_tool.return_value = self.mock_search_tool
        mock_outline_tool.return_value = self.mock_outline_tool

        # Configure default behaviors
        self.mock_ai_generator.generate_response.return_value = "AI response"
        self.mock_session_manager.get_conversation_history.return_value = None
        self.mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"},
            {"name": "get_course_outline"}
        ]
        self.mock_tool_manager.get_last_sources.return_value = []

        # Create config mock
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "/tmp/test_chroma"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "test-model"
        self.mock_config.MAX_HISTORY = 2

        # Import and instantiate RAGSystem
        from rag_system import RAGSystem
        self.rag_system = RAGSystem(self.mock_config)

    def test_passes_tools_to_ai_generator(self):
        """Both tools should be passed to ai_generator"""
        # Act
        self.rag_system.query("test query", session_id=None)

        # Assert
        call_kwargs = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIn("tools", call_kwargs)
        self.assertEqual(len(call_kwargs["tools"]), 2)

    def test_passes_tool_manager(self):
        """tool_manager parameter should be set"""
        # Act
        self.rag_system.query("test query", session_id=None)

        # Assert
        call_kwargs = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIn("tool_manager", call_kwargs)
        self.assertIsNotNone(call_kwargs["tool_manager"])

    def test_retrieves_conversation_history(self):
        """Session history should be fetched when session_id provided"""
        # Arrange
        self.mock_session_manager.get_conversation_history.return_value = "User: Hi\nAssistant: Hello"

        # Act
        self.rag_system.query("test query", session_id="session_1")

        # Assert
        self.mock_session_manager.get_conversation_history.assert_called_with("session_1")
        call_kwargs = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_kwargs["conversation_history"], "User: Hi\nAssistant: Hello")

    def test_returns_response_and_sources(self):
        """Returns tuple of (response, sources)"""
        # Arrange
        self.mock_ai_generator.generate_response.return_value = "The answer is 42"
        self.mock_tool_manager.get_last_sources.return_value = ["Source 1", "Source 2"]

        # Act
        response, sources = self.rag_system.query("test query")

        # Assert
        self.assertEqual(response, "The answer is 42")
        self.assertEqual(sources, ["Source 1", "Source 2"])

    def test_resets_sources_after_retrieval(self):
        """Sources should be cleared after being retrieved"""
        # Act
        self.rag_system.query("test query")

        # Assert
        self.mock_tool_manager.reset_sources.assert_called_once()

    def test_updates_conversation_history(self):
        """Exchange should be added to session when session_id provided"""
        # Arrange
        self.mock_ai_generator.generate_response.return_value = "AI answer"

        # Act
        self.rag_system.query("user question", session_id="session_1")

        # Assert
        self.mock_session_manager.add_exchange.assert_called_once()
        call_args = self.mock_session_manager.add_exchange.call_args[0]
        self.assertEqual(call_args[0], "session_1")
        self.assertIn("user question", call_args[1])
        self.assertEqual(call_args[2], "AI answer")


class TestRAGSystemErrorPropagation(unittest.TestCase):
    """Tests for error handling in RAGSystem"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    def setUp(self, mock_outline_tool, mock_search_tool, mock_tool_manager_class,
              mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Set up test fixtures"""
        # Create mock instances
        self.mock_vector_store = Mock()
        self.mock_ai_generator = Mock()
        self.mock_session_manager = Mock()
        self.mock_tool_manager = Mock()

        # Configure mock classes
        mock_vector_store.return_value = self.mock_vector_store
        mock_ai_generator.return_value = self.mock_ai_generator
        mock_session_manager.return_value = self.mock_session_manager
        mock_tool_manager_class.return_value = self.mock_tool_manager

        # Default behaviors
        self.mock_session_manager.get_conversation_history.return_value = None
        self.mock_tool_manager.get_tool_definitions.return_value = []
        self.mock_tool_manager.get_last_sources.return_value = []

        # Create config
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "/tmp/test_chroma"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "test-model"
        self.mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        self.rag_system = RAGSystem(self.mock_config)

    def test_vector_store_error_propagates(self):
        """Error from vector store flows through tool to response"""
        # Arrange - AI generator returns error message that came from tool
        self.mock_ai_generator.generate_response.return_value = "Search error: connection failed"

        # Act
        response, sources = self.rag_system.query("test query")

        # Assert - error message is returned to caller
        self.assertIn("error", response.lower())

    def test_tool_execution_error_handled(self):
        """Tool execution errors should be handled gracefully"""
        # Arrange - simulate API error
        self.mock_ai_generator.generate_response.side_effect = Exception("API error")

        # Act & Assert - should raise but with useful message
        with self.assertRaises(Exception) as context:
            self.rag_system.query("test query")

        self.assertIn("API error", str(context.exception))


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for complete query flow"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_manager, mock_ai_generator,
                                   mock_vector_store, mock_doc_processor):
        """Query works correctly without a session ID"""
        # Arrange
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response without session"
        mock_ai_generator.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_manager.return_value = mock_session

        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "/tmp/test"
        mock_config.EMBEDDING_MODEL = "model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "key"
        mock_config.ANTHROPIC_MODEL = "model"
        mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Act
        response, sources = rag.query("test query", session_id=None)

        # Assert
        self.assertEqual(response, "Response without session")
        mock_session.add_exchange.assert_not_called()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_builds_correct_prompt(self, mock_session_manager, mock_ai_generator,
                                         mock_vector_store, mock_doc_processor):
        """Query wraps user query in expected prompt format"""
        # Arrange
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_generator.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_manager.return_value = mock_session

        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "/tmp/test"
        mock_config.EMBEDDING_MODEL = "model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "key"
        mock_config.ANTHROPIC_MODEL = "model"
        mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Act
        rag.query("What is machine learning?")

        # Assert
        call_kwargs = mock_ai.generate_response.call_args[1]
        self.assertIn("What is machine learning?", call_kwargs["query"])
        self.assertIn("course materials", call_kwargs["query"])


if __name__ == "__main__":
    unittest.main()
