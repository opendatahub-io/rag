import requests
import json
from typing import Dict, Any, Optional
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger

MODEL_INSTRUCTIONS = """
    You are a helpful assistant with access to financial data through MCP tools.

    IMPORTANT: Transaction all data is from 2025.

    When asked questions, use available tools to find the answer. Follow these rules:

    1. Decide on the tool to use immediately without asking for confirmation
    2. If you need additional information, search for it using whatever details are provided
    3. Chain tool calls as needed - use results from one call as inputs to the next
    4. If one approach doesn't work, try alternative methods silently
    5. Do not narrate your process, explain failures, or describe what you're trying - just do it
    6. Only provide output when you have the final answer
    7. If you truly cannot find the information after multiple attempts, simply state what you were unable to find
    8. Only use the file_search tool IF the questions are related to knowledge base or FAQs. OR when the question is not about transactions or user-specific data.

    Just execute tool calls until you have an answer, then provide it.
"""

class ResponseService:
    def __init__(self, url, model, vector_store_name: str, mcp_url: str):
        self.base_url = url
        self.conversation_history = []
        self.model = model
        # Initialize LlamaStack client
        self.client = LlamaStackClient(base_url=url)
        self.vector_store_name = vector_store_name
        self.mcp_url = mcp_url
    
    def create_session(self) -> str:
        """Create a new agent session using LlamaStack client"""
        try:
            session = self.agent.create_session(session_name=f"voice_session_{self.agent_id[:8]}")
            self.session_id = session.id if hasattr(session, 'id') else str(session)
            return self.session_id
        except Exception as e:
            print(f"Failed to create session: {e}")
            return None
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def get_vector_store(self, vector_store_name: str) -> str:
        """Get the vector store ID"""
        vector_stores = self.client.vector_stores.list()
        vector_store = next(
            (s for s in vector_stores.data if s.name == vector_store_name), None
        )
        print(f"Vector store ID: {vector_store.id}")
        return vector_store.id

    def invoke(self, prompt: str, model_instructions: str) -> Dict[str, Any]:
        if model_instructions == "":
            model_instructions = MODEL_INSTRUCTIONS
        print(model_instructions)
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        try: 
            resp_stream = self.client.responses.create(
                model=self.model,
                instructions=model_instructions,
                tools=[
                    {
                        "type": "mcp",
                        "server_label": "dmcp",
                        "server_description": "MCP Server.",
                        "server_url": f"{self.mcp_url}",
                        "require_approval": "never",
                    },
                    {"type": "file_search", "vector_store_ids": [self.get_vector_store(self.vector_store_name)]},
                ],
                input=prompt,
                stream=True,
            )
            
            # Process the streaming response to extract the full text
            final_text = None
            completed_response = None
            
            for chunk in resp_stream:
                # Check for completed response event - this has the full response object
                if hasattr(chunk, 'type') and chunk.type == 'response.completed':
                    if hasattr(chunk, 'response') and chunk.response:
                        completed_response = chunk.response
                
                # Check for content_part.done events which contain the full text for each part
                if hasattr(chunk, 'type') and chunk.type == 'response.content_part.done':
                    if hasattr(chunk, 'part') and chunk.part:
                        part = chunk.part
                        # The part has a text attribute that is the full text string
                        if hasattr(part, 'text') and part.text:
                            if isinstance(part.text, str):
                                final_text = part.text
                            elif hasattr(part.text, 'text') and part.text.text:
                                final_text = part.text.text
            
            # Extract text from completed response if we haven't found it yet
            if not final_text and completed_response:
                if hasattr(completed_response, 'output') and completed_response.output:
                    for output_item in completed_response.output:
                        # Look for message type output items with content
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                # Check if it's an output_text type with text attribute
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    if hasattr(content_item, 'text') and content_item.text:
                                        if isinstance(content_item.text, str):
                                            final_text = content_item.text
                                            break
                                        elif hasattr(content_item.text, 'text'):
                                            final_text = content_item.text.text
                                            break
                            if final_text:
                                break
            
            if final_text:
                self.conversation_history.append({"role": "assistant", "content": final_text})
                return {"output": final_text}
            else:
                return {"output": "No response text found in stream"}
                
        except Exception as e:
            print(f"Response error: {e}")
            return {"output": f"Response invocation failed: {str(e)}", "error": "invocation_failed"}
