import subprocess
import json
import os

class MCPClient:
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path

    async def read_file(self, path: str, auth_token: str) -> str:
        """
        Calls the MCP server to read a file.
        Uses subprocess to run the MCP server in stdio mode (simplified for this demo).
        In a real scenario, this would connect to a persistent MCP server process.
        """
        # Note: This is a simplified "one-shot" call for demonstration.
        # A real MCP client would maintain a persistent connection.
        
        # We will use the 'mcp' CLI or just invoke the script directly if it supports it.
        # For this mock, we'll import the server function directly to simulate the RPC 
        # because running a full stdio subprocess for every call in this dev env might be flaky without full async plumbing.
        
        # SIMULATION MODE: Import the server code directly
        try:
            from mcp_server.server import read_file
            return read_file(path, auth_token)
        except ImportError:
            return "Error: Could not import MCP server"
        except Exception as e:
            return f"Error: {str(e)}"

    async def list_files(self, auth_token: str, folder: str = "") -> dict:
        """
        Calls the MCP server to list files in a folder.
        """
        try:
            from mcp_server.server import list_files
            result = list_files(auth_token, folder)
            return json.loads(result)
        except ImportError:
            return {"error": "Could not import MCP server"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from MCP server"}
        except Exception as e:
            return {"error": str(e)}

    async def write_file(self, path: str, content: str, auth_token: str) -> dict:
        """
        Calls the MCP server to write/upload a file.
        """
        try:
            from mcp_server.server import write_file
            result = write_file(path, content, auth_token)
            return json.loads(result)
        except ImportError:
            return {"error": "Could not import MCP server"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from MCP server"}
        except Exception as e:
            return {"error": str(e)}

    async def delete_file(self, path: str, auth_token: str) -> dict:
        """
        Calls the MCP server to delete a file.
        """
        try:
            from mcp_server.server import delete_file
            result = delete_file(path, auth_token)
            return json.loads(result)
        except ImportError:
            return {"error": "Could not import MCP server"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from MCP server"}
        except Exception as e:
            return {"error": str(e)}

# Singleton instance
SERVER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mcp_server", "server.py")
mcp_client = MCPClient(SERVER_PATH)

