"""Tests for VectorStore class and configuration"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVectorStoreMaxResults(unittest.TestCase):
    """Tests for VectorStore max_results configuration"""

    @patch("vector_store.chromadb")
    @patch("vector_store.SentenceTransformer")
    def test_max_results_stored_correctly(self, mock_transformer, mock_chromadb):
        """VectorStore stores max_results parameter"""
        from vector_store import VectorStore

        # Act
        store = VectorStore("/tmp/test", "test-model", max_results=5)

        # Assert
        self.assertEqual(store.max_results, 5)

    @patch("vector_store.chromadb")
    @patch("vector_store.SentenceTransformer")
    def test_max_results_zero_is_invalid(self, mock_transformer, mock_chromadb):
        """MAX_RESULTS of 0 would return no results - this catches the config bug"""
        from vector_store import VectorStore

        # Act
        store = VectorStore("/tmp/test", "test-model", max_results=0)

        # Assert - max_results of 0 is problematic!
        self.assertEqual(store.max_results, 0)
        # This test documents the bug: when max_results=0, no results are returned

    @patch("vector_store.chromadb")
    @patch("vector_store.SentenceTransformer")
    def test_search_uses_max_results(self, mock_transformer, mock_chromadb):
        """Search method uses max_results when limit not specified"""
        from vector_store import VectorStore

        # Arrange
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore("/tmp/test", "test-model", max_results=5)

        # Act
        store.search(query="test query")

        # Assert - should use max_results (5) as n_results
        call_kwargs = mock_collection.query.call_args[1]
        self.assertEqual(call_kwargs["n_results"], 5)

    @patch("vector_store.chromadb")
    @patch("vector_store.SentenceTransformer")
    def test_search_with_zero_max_results_returns_nothing(
        self, mock_transformer, mock_chromadb
    ):
        """When max_results=0, search returns empty results"""
        from vector_store import VectorStore

        # Arrange
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore("/tmp/test", "test-model", max_results=0)

        # Act
        store.search(query="test query")

        # Assert - n_results=0 means no results returned!
        call_kwargs = mock_collection.query.call_args[1]
        self.assertEqual(call_kwargs["n_results"], 0)  # This is the bug!


class TestConfigMaxResults(unittest.TestCase):
    """Tests for Config MAX_RESULTS setting"""

    def test_config_max_results_should_be_positive(self):
        """Config MAX_RESULTS must be > 0 for search to work"""
        from config import Config

        config = Config()

        # This test FAILS with current config (MAX_RESULTS=0)
        # It documents the bug that needs fixing
        self.assertGreater(
            config.MAX_RESULTS,
            0,
            f"MAX_RESULTS is {config.MAX_RESULTS}, but must be > 0 for search to return results!",
        )


class TestVectorStoreSearch(unittest.TestCase):
    """Tests for VectorStore.search() method"""

    @patch("vector_store.chromadb")
    @patch("vector_store.SentenceTransformer")
    def test_search_with_explicit_limit_overrides_max_results(
        self, mock_transformer, mock_chromadb
    ):
        """Explicit limit parameter overrides max_results"""
        from vector_store import VectorStore

        # Arrange
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore("/tmp/test", "test-model", max_results=5)

        # Act - use explicit limit
        store.search(query="test", limit=10)

        # Assert - should use limit (10), not max_results (5)
        call_kwargs = mock_collection.query.call_args[1]
        self.assertEqual(call_kwargs["n_results"], 10)


if __name__ == "__main__":
    unittest.main()
