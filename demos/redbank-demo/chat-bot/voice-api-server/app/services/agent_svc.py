import requests
import json
from typing import Dict, Any, Optional
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger

class AgentService:
    def __init__(self, url, route, name, api_key, model, vector_store_name: str):
        self.base_url = url
        self.agent_route = route
        self.agent_id = name
        self.api_key = api_key
        self.session_id = None
        self.conversation_history = []
        self.model = model
        # Initialize LlamaStack client
        self.client = LlamaStackClient(base_url=url)
        self.agent = Agent(
            self.client,
            model=model,
            instructions = """You are a banking assistant; Use the knowledge tool to answer questions anduse the MCP tools to fetch user banking information by phone number. Make multiple tool calls to get complete account details including statements and transactions. Do not retrive info not asked by the user. Always use the phone +353 85 148 0072. If no answer is found, say so directly""",

            tools=[
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_description": "MCP Server.",
                    "server_url": "http://redbank-mcp-server:8000/mcp"
                },
                {
                    # "name": "builtin::rag/knowledge_search",
                    "type": "file_search",
                    "vector_store_ids": [self.get_vector_store(vector_store_name)],
                }
            ],
        )
    
    def create_session(self) -> str:
        """Create a new agent session using LlamaStack client"""
        try:
            session = self.agent.create_session(session_name=f"voice_session_{self.agent_id[:8]}")
            self.session_id = session.id if hasattr(session, 'id') else str(session)
            return self.session_id
        except Exception as e:
            print(f"Failed to create session: {e}")
            return None
    
    def invoke(self, text: str, tools_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with the given text using LlamaStack client"""
        # Create a session if we don't have one
        if not self.session_id:
            self.session_id = self.create_session()
        
        if not self.session_id:
            return {"output": "Failed to create session", "error": "session_failed"}
        
        # Add user message to conversation history
        # self.conversation_history.append({"role": "user", "content": text})
        
        try:
            # Use LlamaStack client to create turn
            response_stream = self.agent.create_turn(
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                session_id=self.session_id,
                stream=True
            )

            
            # Process the streaming response to get the final answer
            final_response = None
            
            for chunk in response_stream:
                # Method 1: Check for TurnCompleted event with final_text
                if hasattr(chunk, 'event') and chunk.event:
                    event = chunk.event
                    event_type = type(event).__name__
                    
                    # TurnCompleted event has final_text attribute
                    if event_type == 'TurnCompleted' and hasattr(event, 'final_text') and event.final_text:
                        final_response = event.final_text
                        break
                
                # Method 2: Check chunk.response for ResponseObject
                if hasattr(chunk, 'response') and chunk.response:
                    response = chunk.response
                    
                    # ResponseObject has output which is a list of messages
                    if hasattr(response, 'output') and response.output:
                        for output_item in response.output:
                            # Check if it's a message with content
                            if hasattr(output_item, 'content') and output_item.content:
                                # content is a list of content items
                                for content_item in output_item.content:
                                    if hasattr(content_item, 'text') and content_item.text:
                                        final_response = content_item.text
                                        break
                                if final_response:
                                    break
                    
                    # Also check for text attribute directly on response
                    if not final_response and hasattr(response, 'text') and response.text:
                        # text might be an object with format, check if it has a value
                        text_obj = response.text
                        if hasattr(text_obj, 'text') and text_obj.text:
                            final_response = text_obj.text
                        elif isinstance(text_obj, str):
                            final_response = text_obj
                    
                    if final_response:
                        break
            
            if final_response:
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": final_response})
                return {"output": final_response, "model": "vllm-inference/llama-4-scout-17b-16e-w4a16"}
            
            return {"output": "No response from agent"}
        except Exception as e:
            print(f"Agent error: {e}")
            return {"output": f"Agent invocation failed: {str(e)}", "error": "invocation_failed"}
    
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
