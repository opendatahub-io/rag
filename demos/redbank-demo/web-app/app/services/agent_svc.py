import requests
import json
from typing import Dict, Any, Optional
from llama_stack_client import LlamaStackClient, Agent

class AgentService:
    def __init__(self, url, route, name, api_key, model):
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

            # tools=[
            #     "mcp::redbank-financials", 
            #     {
            #         "name": "builtin::rag/knowledge_search",
            #         "args": {"vector_db_ids": ['vs_1f1dd1b7-49ad-4ceb-8e8d-f0bf9afe2179']},
            #     }
            # ],
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
                if hasattr(chunk, 'event') and chunk.event:
                    if hasattr(chunk.event, 'payload') and chunk.event.payload:
                        payload = chunk.event.payload
                        
                        # Check if this is the turn_complete event with the final response
                        if (hasattr(payload, 'event_type') and 
                            payload.event_type == 'turn_complete' and
                            hasattr(payload, 'turn') and payload.turn):
                            
                            # Extract the content from the turn
                            turn = payload.turn
                            
                            # Try different ways to extract content
                            if hasattr(turn, 'output_message') and turn.output_message:
                                output_message = turn.output_message
                                
                                if hasattr(output_message, 'content') and output_message.content:
                                    final_response = output_message.content
                                    break
                                elif hasattr(output_message, 'text') and output_message.text:
                                    final_response = output_message.text
                                    break
                            
                            # # Try direct content access
                            # if hasattr(turn, 'content') and turn.content:
                            #     final_response = turn.content
                            #     break
            
            if final_response:
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": final_response})
                return {"output": final_response, "model": "vllm-inference/llama-4-scout-17b-16e-w4a16"}
            
            return {"output": "No response from agent"}
            print(final_response)
        except Exception as e:
            print(f"Agent error: {e}")
            return {"output": f"Agent invocation failed: {str(e)}", "error": "invocation_failed"}
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []