"""Shared pytest fixtures for RAG system tests"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "/tmp/test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore instance"""
    store = Mock()
    store.search.return_value = []
    store.get_all_courses.return_value = []
    store.add_course.return_value = None
    store.add_chunks.return_value = None
    return store


@pytest.fixture
def mock_ai_generator():
    """Create a mock AIGenerator instance"""
    generator = Mock()
    generator.generate_response.return_value = "Mock AI response"
    return generator


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager instance"""
    manager = Mock()
    manager.create_session.return_value = "test-session-id"
    manager.get_conversation_history.return_value = None
    manager.add_exchange.return_value = None
    manager.clear_session.return_value = None
    return manager


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager instance"""
    manager = Mock()
    manager.get_tool_definitions.return_value = [
        {"name": "search_course_content"},
        {"name": "get_course_outline"}
    ]
    manager.get_last_sources.return_value = []
    manager.reset_sources.return_value = None
    return manager


@pytest.fixture
def mock_rag_system(mock_config, mock_vector_store, mock_ai_generator,
                    mock_session_manager, mock_tool_manager):
    """Create a mock RAGSystem with all dependencies mocked"""
    rag = Mock()
    rag.config = mock_config
    rag.vector_store = mock_vector_store
    rag.ai_generator = mock_ai_generator
    rag.session_manager = mock_session_manager
    rag.tool_manager = mock_tool_manager

    # Configure query method
    rag.query.return_value = ("Mock response", ["Source 1"])

    # Configure get_course_analytics method
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }

    return rag


@pytest.fixture
def sample_course_data():
    """Sample course document data for testing"""
    return {
        "title": "Introduction to Python",
        "link": "https://example.com/python-course",
        "instructor": "Test Instructor",
        "lessons": [
            {
                "title": "Lesson 1: Getting Started",
                "link": "https://example.com/lesson1",
                "content": "Python is a high-level programming language..."
            },
            {
                "title": "Lesson 2: Variables and Types",
                "link": "https://example.com/lesson2",
                "content": "Variables in Python are created when you assign a value..."
            }
        ]
    }


@pytest.fixture
def sample_chunks():
    """Sample text chunks for vector store testing"""
    return [
        {
            "id": "chunk_1",
            "text": "Python is a high-level programming language known for its simplicity.",
            "metadata": {
                "course_title": "Introduction to Python",
                "lesson_title": "Getting Started"
            }
        },
        {
            "id": "chunk_2",
            "text": "Variables in Python are dynamically typed.",
            "metadata": {
                "course_title": "Introduction to Python",
                "lesson_title": "Variables and Types"
            }
        }
    ]


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is Python?",
        "session_id": None
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data"""
    return {
        "answer": "Python is a high-level programming language...",
        "sources": ["Introduction to Python - Getting Started"],
        "session_id": "test-session-id"
    }
