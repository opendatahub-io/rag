# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any
from llama_stack_client import LlamaStackClient

MODEL_INSTRUCTIONS = """
    /no_think
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
            session = self.agent.create_session(
                session_name=f"voice_session_{self.agent_id[:8]}"
            )
            self.session_id = session.id if hasattr(session, "id") else str(session)
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
            resp = self.client.responses.create(
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
                    {
                        "type": "file_search",
                        "vector_store_ids": [
                            self.get_vector_store(self.vector_store_name)
                        ],
                    },
                ],
                input=self.conversation_history,
                stream=False,
            )

            self.conversation_history.append(
                {"role": "assistant", "content": resp.output_text}
            )
            return {"output": resp.output_text}

        except Exception as e:
            print(f"Response error: {e}")
            return {
                "output": f"Response invocation failed: {str(e)}",
                "error": "invocation_failed",
            }
