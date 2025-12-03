from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class MCPReadFileInput(BaseModel):
    """Input schema for MCPReadFile"""
    path: str = Field(..., description="The file path to read (e.g., 'input.json')")
    auth_token: str = Field(..., description="Authentication token for MCP server")

class MCPReadFileTool(BaseTool):
    name: str = "read_file_from_drive"
    description: str = "Reads a file from the Google Drive (via MCP). Use this to access input.json and output.json files."
    args_schema: Type[BaseModel] = MCPReadFileInput

    def _run(self, path: str, auth_token: str) -> str:
        """Execute the tool to read a file"""
        try:
            from mcp_server.server import read_file
            content = read_file(path, auth_token)
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
