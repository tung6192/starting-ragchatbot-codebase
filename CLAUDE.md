# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv` to run Python commands. Do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Run the server (from project root)
./run.sh

# Run the server manually (from backend/)
cd backend && uv run uvicorn app:app --reload --port 8000

# Run with external access
cd backend && uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot with a FastAPI backend and vanilla JS frontend.

### Query Flow

```
User Query → Frontend (script.js)
    → POST /api/query
    → app.py → RAGSystem.query()
    → AIGenerator calls Claude API with tools
    → Claude decides: direct answer OR use search_course_content tool
    → If tool use: CourseSearchTool → VectorStore (ChromaDB) → return to Claude
    → Final response returned to frontend
```

### Backend Components (backend/)

- **app.py**: FastAPI endpoints (`/api/query`, `/api/courses`), serves static frontend
- **rag_system.py**: Orchestrates document processing, vector search, and AI generation
- **ai_generator.py**: Claude API integration with tool-use support; makes two API calls when tools are used
- **vector_store.py**: ChromaDB with two collections: `course_catalog` (metadata) and `course_content` (chunks)
- **document_processor.py**: Parses course files, extracts metadata, chunks text (~800 chars with 100 overlap)
- **search_tools.py**: Defines `CourseSearchTool` that Claude can invoke; `ToolManager` handles execution
- **session_manager.py**: Maintains conversation history per session (default: last 2 exchanges)
- **config.py**: Loads `.env` from project root, defines chunk sizes, model settings

### Frontend (frontend/)

Vanilla JS with marked.js for markdown rendering. Communicates via `/api/query` and `/api/courses`.

### Document Format (docs/)

Course documents follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]
```

### Key Configuration (config.py)

- Model: `claude-sonnet-4-20250514`
- Embedding: `all-MiniLM-L6-v2`
- Chunk size: 800 chars, overlap: 100 chars
- Max search results: 5
- Conversation history: 2 exchanges
