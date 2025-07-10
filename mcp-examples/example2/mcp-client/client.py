import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from anthropic import Anthropic
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here
    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server via streamable HTTP

        Args:
            server_url: URL of the HTTP MCP server (e.g., http://localhost:8000)
        """
        # Ensure the URL has the /mcp path
        if '/mcp' not in server_url:
            if server_url.endswith('/'):
                server_url = server_url + 'mcp'
            else:
                server_url = server_url + '/mcp'
        print(f"Connecting to server at: {server_url}")
        # Connect to the server using streamable HTTP
        await self.connect_to_http_server(server_url)
            
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
            
    # Removed stdio connection method as we're focusing only on streamable HTTP
    
    async def connect_to_http_server(self, server_url: str):
        """Connect to an MCP server via HTTP using streamable HTTP client

        Args:
            server_url: URL of the HTTP MCP server (e.g., http://localhost:8000)
        """
        try:
            print(f"[DEBUG] Connecting to server at {server_url}")
            # Use streamablehttp_client to connect to the server
            http_transport = await self.exit_stack.enter_async_context(streamablehttp_client(server_url))
            print(f"[DEBUG] Got HTTP transport: {http_transport}")
            self.read_stream, self.write_stream, _ = http_transport
            print(f"[DEBUG] Got read/write streams: {self.read_stream}, {self.write_stream}")
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.read_stream, self.write_stream))
            print(f"[DEBUG] Created client session: {self.session}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to server: {e}")
            raise
        
        # Initialize the session
        await self.session.initialize()
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # Initialize messages with user query
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        #print("\n[DEBUG] Available tools:\n" + str(available_tools))
        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system="Format Kubernetes resource data as clean markdown tables only. No explanations or text outside the table.",
            messages=messages,
            tools=available_tools
        )
        #print("\n[DEBUG] Claude response:\n" + str(response))
        # Process response and handle tool calls
        final_text = []

        # Store the assistant's message content for the conversation history
        assistant_message_content = []
        for content in response.content:
            print("\n[DEBUG] Content type: " + content.type)
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                #print("\n[DEBUG] Tool name: " + tool_name)
                #print("\n[DEBUG] Tool args: " + str(tool_args))
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                #print("\n[DEBUG] Tool result: " + str(result.content))
                
                # Store the raw result for Claude to format
                raw_result = result.content[0].text
                
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": raw_result + "\n\nCRITICAL INSTRUCTION: Parse this JSON and ONLY output a markdown table. DO NOT include ANY text before or after the table. NO explanations, NO descriptions, NO introductions, NO conclusions.\n\nFor pods data, use EXACTLY these column headers:\n| Pod Name | Status | CPU Request | Memory Request |\n\nFor services data, use EXACTLY these column headers:\n| Service Name | Status | Cluster IP | Ports | Node Port | Selector |\n\nYour entire response must be ONLY the markdown table and nothing else."
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system="Format Kubernetes resource data as clean markdown tables only. No explanations or text outside the table.",
                    messages=messages,
                    tools=available_tools
                )
                #print("\n[DEBUG] Response content:\n" + str(response.content))
                final_text.append(response.content[0].text)

        return "\n".join(final_text)    

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()        

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_url>")
        print("Example: python client.py http://localhost:8000")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())        
