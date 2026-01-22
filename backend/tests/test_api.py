"""Tests for FastAPI endpoints"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define Pydantic models (same as app.py)
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# Create a test app without static file mounting
def create_test_app(mock_rag_system):
    """Create a FastAPI test app with mocked RAG system"""
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"status": "cleared", "session_id": session_id}
        except Exception:
            return {"status": "not_found", "session_id": session_id}

    return app


@pytest.fixture
def test_rag_system():
    """Create a mock RAG system for API testing"""
    rag = Mock()

    # Mock session manager
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "test-session-123"
    rag.session_manager.clear_session.return_value = None

    # Mock query method
    rag.query.return_value = (
        "Python is a high-level programming language.",
        ["Introduction to Python - Lesson 1"]
    )

    # Mock analytics
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Advanced Python", "Data Science"]
    }

    return rag


@pytest.fixture
def client(test_rag_system):
    """Create a test client with mocked dependencies"""
    app = create_test_app(test_rag_system)
    return TestClient(app)


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_success(self, client, test_rag_system):
        """Successful query returns answer and sources"""
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Python is a high-level programming language."
        assert data["session_id"] == "test-session-123"

    def test_query_with_session_id(self, client, test_rag_system):
        """Query with existing session ID uses that session"""
        response = client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing-session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"
        test_rag_system.query.assert_called_with("Tell me more", "existing-session")

    def test_query_creates_session_when_none_provided(self, client, test_rag_system):
        """Query without session_id creates a new session"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        test_rag_system.session_manager.create_session.assert_called_once()

    def test_query_returns_sources(self, client, test_rag_system):
        """Query response includes source references"""
        response = client.post(
            "/api/query",
            json={"query": "Explain variables"}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_query_missing_query_field(self, client):
        """Request without query field returns 422"""
        response = client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == 422

    def test_query_empty_query(self, client, test_rag_system):
        """Empty query string is accepted (RAG system handles it)"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty query is valid per Pydantic model
        assert response.status_code == 200

    def test_query_error_returns_500(self, client, test_rag_system):
        """Internal error returns 500 status code"""
        test_rag_system.query.side_effect = Exception("Database connection failed")

        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, client, test_rag_system):
        """Successful request returns course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3

    def test_get_courses_returns_correct_titles(self, client, test_rag_system):
        """Course titles are returned correctly"""
        response = client.get("/api/courses")

        data = response.json()
        assert "Python Basics" in data["course_titles"]
        assert "Advanced Python" in data["course_titles"]
        assert "Data Science" in data["course_titles"]

    def test_get_courses_empty_catalog(self, client, test_rag_system):
        """Empty course catalog returns zero count"""
        test_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_error_returns_500(self, client, test_rag_system):
        """Internal error returns 500 status code"""
        test_rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]


class TestSessionEndpoint:
    """Tests for DELETE /api/session/{session_id} endpoint"""

    def test_clear_session_success(self, client, test_rag_system):
        """Clearing existing session returns success status"""
        response = client.delete("/api/session/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "test-session-123"

    def test_clear_session_calls_manager(self, client, test_rag_system):
        """Clear session delegates to session manager"""
        client.delete("/api/session/my-session")

        test_rag_system.session_manager.clear_session.assert_called_with("my-session")

    def test_clear_nonexistent_session(self, client, test_rag_system):
        """Clearing nonexistent session returns not_found status"""
        test_rag_system.session_manager.clear_session.side_effect = Exception("Not found")

        response = client.delete("/api/session/nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"


class TestRequestValidation:
    """Tests for request validation and edge cases"""

    def test_query_with_special_characters(self, client, test_rag_system):
        """Query with special characters is handled"""
        response = client.post(
            "/api/query",
            json={"query": "What is <script>alert('xss')</script>?"}
        )

        assert response.status_code == 200
        # The query is passed through; output sanitization is separate
        test_rag_system.query.assert_called()

    def test_query_with_unicode(self, client, test_rag_system):
        """Query with unicode characters is handled"""
        response = client.post(
            "/api/query",
            json={"query": "What is Python? \u4e2d\u6587\u6d4b\u8bd5"}
        )

        assert response.status_code == 200

    def test_query_with_long_text(self, client, test_rag_system):
        """Very long query is accepted"""
        long_query = "What is Python? " * 100

        response = client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 200

    def test_invalid_json_body(self, client):
        """Invalid JSON returns 422"""
        response = client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        """Wrong content type is rejected"""
        response = client.post(
            "/api/query",
            content="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422


class TestResponseFormat:
    """Tests for response format consistency"""

    def test_query_response_schema(self, client, test_rag_system):
        """Query response matches expected schema"""
        response = client.post(
            "/api/query",
            json={"query": "test"}
        )

        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_schema(self, client, test_rag_system):
        """Courses response matches expected schema"""
        response = client.get("/api/courses")

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_session_clear_response_schema(self, client, test_rag_system):
        """Session clear response matches expected schema"""
        response = client.delete("/api/session/test")

        data = response.json()
        assert "status" in data
        assert "session_id" in data
        assert data["status"] in ["cleared", "not_found"]
