import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.http import http_client

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
    async def connect_to_server(self, server_path_or_url: str):
        """Connect to an MCP server either via stdio or HTTP

        Args:
            server_path_or_url: Path to the server script (.py or .js) or HTTP URL
        """
        # Check if it's an HTTP URL
        if server_path_or_url.startswith("http://") or server_path_or_url.startswith("https://"):
            await self.connect_to_http_server(server_path_or_url)
        else:
            await self.connect_to_stdio_server(server_path_or_url)
            
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
            
    async def connect_to_stdio_server(self, server_script_path: str):
        """Connect to an MCP server via stdio

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python3" if is_python else "node"  # Using python3 instead of python
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        # Initialize the session
        await self.session.initialize()
        
    async def connect_to_http_server(self, server_url: str):
        """Connect to an MCP server via HTTP

        Args:
            server_url: URL of the HTTP MCP server (e.g., http://localhost:8000)
        """
        from mcp.client.http import http_client
        
        http_transport = await self.exit_stack.enter_async_context(http_client(server_url))
        self.session = await self.exit_stack.enter_async_context(ClientSession(http_transport))
        
        # Initialize the session
        await self.session.initialize()

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
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
            messages=messages,
            tools=available_tools
        )
        #print("\n[DEBUG] Claude response:\n" + str(response))
        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            #print("\n[DEBUG] Content type: " + content.type)
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
                print("\nTABULAR RESPONSE")
                data = json.loads(result.content[0].text)
                if "pods" in data:
                    table_data = [[p["pod_name"], p["status"], p["cpu_request"], p["memory_request"]] for p in data["pods"]]
                    print(tabulate(table_data, headers=["Pod Name", "Status", "CPU Request", "Memory Request"], tablefmt="grid"))
                elif "error" in data:
                    print(data["error"])
                else:
                    print(data["message"])               
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
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
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
        print("Usage: python client.py <path_to_server_script>")
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