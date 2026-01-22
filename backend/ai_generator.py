import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
2. **get_course_outline**: Get a course's complete structure including title, link, and all lessons

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - Course structure or outline
  - List of lessons in a course
  - What topics a course covers
  - Course link or overview information
  - When asked to list the lessons
- Use **search_course_content** for questions about:
  - Specific content or concepts within courses
  - Detailed educational materials
  - How-to questions about course topics
- You may make up to 2 sequential tool calls when needed for complex queries
- Use multiple calls for: comparisons between courses, multi-part questions, or when one tool's result informs the next search
- After each tool result, decide if you have enough information to answer
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol for Outline Queries:
- Always include the course title and course link in your response
- List all lessons with their lesson number and lesson title
- Format the lesson list clearly (e.g., numbered list)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(
                response=response,
                messages=api_params["messages"].copy(),
                system=api_params["system"],
                tools=tools,
                tool_manager=tool_manager,
                current_round=1
            )
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, response, messages: List,
                               system: str, tools: Optional[List],
                               tool_manager, current_round: int = 1) -> str:
        """
        Handle execution of tool calls with support for sequential rounds.

        Args:
            response: The response containing tool use requests
            messages: Current message history
            system: System prompt
            tools: Available tools for subsequent calls
            tool_manager: Manager to execute tools
            current_round: Current tool round (1-indexed)

        Returns:
            Final response text after tool execution
        """
        # Copy messages to avoid mutation
        messages = messages.copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Determine if tools should be included in next call
        include_tools = current_round < self.MAX_TOOL_ROUNDS

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system
        }

        # Include tools if more rounds are allowed
        if include_tools and tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get next response
        next_response = self.client.messages.create(**api_params)

        # Check if another tool round is needed and allowed
        if next_response.stop_reason == "tool_use" and current_round < self.MAX_TOOL_ROUNDS:
            return self._handle_tool_execution(
                response=next_response,
                messages=messages,
                system=system,
                tools=tools,
                tool_manager=tool_manager,
                current_round=current_round + 1
            )

        return next_response.content[0].text