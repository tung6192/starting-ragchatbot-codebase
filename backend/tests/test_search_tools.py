"""Tests for CourseSearchTool and ToolManager classes"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager, Tool
from vector_store import SearchResults


class TestCourseSearchToolExecute(unittest.TestCase):
    """Tests for CourseSearchTool.execute() method"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_calls_vector_store_with_correct_params(self):
        """Verify parameters are passed correctly to vector store"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=["content"], metadata=[{"course_title": "Test"}], distances=[0.5]
        )
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        self.search_tool.execute(
            query="test query",
            course_name="Test Course",
            lesson_number=3
        )

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=3
        )

    def test_execute_returns_error_from_search_results(self):
        """When SearchResults.error is set, return the error message"""
        # Arrange
        error_message = "No course found matching 'NonExistent'"
        self.mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=error_message
        )

        # Act
        result = self.search_tool.execute(query="test query", course_name="NonExistent")

        # Assert
        self.assertEqual(result, error_message)

    def test_execute_handles_empty_results_no_filters(self):
        """Returns 'No relevant content found.' when empty with no filters"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        # Act
        result = self.search_tool.execute(query="obscure query")

        # Assert
        self.assertEqual(result, "No relevant content found.")

    def test_execute_handles_empty_results_with_course_filter(self):
        """Returns message including course name when empty with course filter"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        # Act
        result = self.search_tool.execute(query="test", course_name="Python 101")

        # Assert
        self.assertIn("Python 101", result)
        self.assertIn("No relevant content found", result)

    def test_execute_handles_empty_results_with_lesson_filter(self):
        """Returns message including lesson number when empty with lesson filter"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        # Act
        result = self.search_tool.execute(query="test", lesson_number=5)

        # Assert
        self.assertIn("lesson 5", result)
        self.assertIn("No relevant content found", result)

    def test_execute_formats_results_correctly(self):
        """Verify formatting with headers and sources"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=["Content about Python basics"],
            metadata=[{"course_title": "Python Course", "lesson_number": 1}],
            distances=[0.3]
        )
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.search_tool.execute(query="python basics")

        # Assert
        self.assertIn("[Python Course - Lesson 1]", result)
        self.assertIn("Content about Python basics", result)

    def test_format_results_includes_lesson_links(self):
        """Sources should include markdown links when lesson links are available"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 2}],
            distances=[0.2]
        )
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"

        # Act
        self.search_tool.execute(query="test")

        # Assert
        self.assertIn("https://example.com/lesson2", self.search_tool.last_sources[0])
        self.assertIn("[Test Course - Lesson 2]", self.search_tool.last_sources[0])

    def test_format_results_deduplicates_sources(self):
        """No duplicate sources should appear in last_sources"""
        # Arrange
        self.mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Same Course", "lesson_number": 1},
                {"course_title": "Same Course", "lesson_number": 1}  # Duplicate
            ],
            distances=[0.2, 0.3]
        )
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        self.search_tool.execute(query="test")

        # Assert - should only have one source even though two results
        self.assertEqual(len(self.search_tool.last_sources), 1)


class TestToolManager(unittest.TestCase):
    """Tests for ToolManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()

        # Create a mock tool
        self.mock_tool = Mock(spec=Tool)
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool"
        }
        self.mock_tool.execute.return_value = "Tool executed successfully"
        self.mock_tool.last_sources = ["source1", "source2"]

    def test_execute_tool_routes_to_correct_tool(self):
        """Verify the correct tool is invoked"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        result = self.tool_manager.execute_tool("test_tool", param1="value1")

        # Assert
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        self.assertEqual(result, "Tool executed successfully")

    def test_execute_tool_not_found(self):
        """Returns error message when tool is not found"""
        # Act
        result = self.tool_manager.execute_tool("nonexistent_tool")

        # Assert
        self.assertIn("not found", result)
        self.assertIn("nonexistent_tool", result)

    def test_get_last_sources(self):
        """Retrieves tool sources correctly"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        sources = self.tool_manager.get_last_sources()

        # Assert
        self.assertEqual(sources, ["source1", "source2"])

    def test_reset_sources(self):
        """Clears all tool sources"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        self.tool_manager.reset_sources()

        # Assert
        self.assertEqual(self.mock_tool.last_sources, [])

    def test_get_tool_definitions(self):
        """Returns tool definitions for all registered tools"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        definitions = self.tool_manager.get_tool_definitions()

        # Assert
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "test_tool")

    def test_register_tool_requires_name(self):
        """Raises error if tool definition has no name"""
        # Arrange
        bad_tool = Mock(spec=Tool)
        bad_tool.get_tool_definition.return_value = {"description": "No name"}

        # Act & Assert
        with self.assertRaises(ValueError):
            self.tool_manager.register_tool(bad_tool)


if __name__ == "__main__":
    unittest.main()
