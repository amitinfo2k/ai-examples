from mcp.server.fastmcp import FastMCP
import os

# Initialize FastMCP server
mcp = FastMCP("GoogleDriveMock")

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage"))

@mcp.tool()
def read_file(path: str, auth_token: str) -> str:
    """
    Reads a file from the mock Google Drive storage.
    
    Args:
        path: The relative path to the file (e.g., 'input.json').
        auth_token: A simulated auth token. For this mock, 'valid_token' is required.
    """
    if auth_token != "valid_token":
        return "Error: Invalid Authentication Token"
    
    # Prevent directory traversal
    # Ensure path is treated as relative to storage root
    path = path.lstrip("/")
    safe_path = os.path.normpath(os.path.join(STORAGE_ROOT, path))
    
    if not safe_path.startswith(STORAGE_ROOT):
        return "Error: Access Denied (Directory Traversal)"
    
    if not os.path.exists(safe_path):
        return f"Error: File not found at {path}"
        
    try:
        with open(safe_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

if __name__ == "__main__":
    mcp.run()
