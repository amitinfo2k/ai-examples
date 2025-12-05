from mcp.server.fastmcp import FastMCP
import os
import json

# Initialize FastMCP server
mcp = FastMCP("GoogleDriveMock")

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage"))

# Ensure storage directory exists
os.makedirs(STORAGE_ROOT, exist_ok=True)


@mcp.tool()
def list_files(auth_token: str, folder: str = "") -> str:
    """
    Lists files in the mock Google Drive storage.
    
    Args:
        auth_token: A simulated auth token. For this mock, 'valid_token' is required.
        folder: Optional subfolder path to list files from.
    """
    if auth_token != "valid_token":
        return json.dumps({"error": "Invalid Authentication Token"})
    
    folder = folder.lstrip("/") if folder else ""
    target_path = os.path.normpath(os.path.join(STORAGE_ROOT, folder))
    
    if not target_path.startswith(STORAGE_ROOT):
        return json.dumps({"error": "Access Denied (Directory Traversal)"})
    
    if not os.path.exists(target_path):
        return json.dumps({"error": f"Folder not found: {folder}"})
    
    try:
        files = []
        for item in os.listdir(target_path):
            # Skip hidden files and swap files
            if item.startswith('.'):
                continue
            
            item_path = os.path.join(target_path, item)
            relative_path = os.path.relpath(item_path, STORAGE_ROOT)
            
            file_info = {
                "name": item,
                "path": relative_path,
                "is_directory": os.path.isdir(item_path),
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0
            }
            files.append(file_info)
        
        return json.dumps({"files": files, "folder": folder or "/"})
    except Exception as e:
        return json.dumps({"error": f"Error listing files: {str(e)}"})


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


@mcp.tool()
def write_file(path: str, content: str, auth_token: str) -> str:
    """
    Writes/uploads a file to the mock Google Drive storage.
    
    Args:
        path: The relative path where to save the file (e.g., 'input.json').
        content: The file content to write.
        auth_token: A simulated auth token. For this mock, 'valid_token' is required.
    """
    if auth_token != "valid_token":
        return json.dumps({"error": "Invalid Authentication Token"})
    
    # Prevent directory traversal
    path = path.lstrip("/")
    safe_path = os.path.normpath(os.path.join(STORAGE_ROOT, path))
    
    if not safe_path.startswith(STORAGE_ROOT):
        return json.dumps({"error": "Access Denied (Directory Traversal)"})
    
    try:
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(safe_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(safe_path, "w") as f:
            f.write(content)
        
        return json.dumps({
            "success": True,
            "path": path,
            "size": len(content)
        })
    except Exception as e:
        return json.dumps({"error": f"Error writing file: {str(e)}"})


@mcp.tool()
def delete_file(path: str, auth_token: str) -> str:
    """
    Deletes a file from the mock Google Drive storage.
    
    Args:
        path: The relative path to the file to delete.
        auth_token: A simulated auth token. For this mock, 'valid_token' is required.
    """
    if auth_token != "valid_token":
        return json.dumps({"error": "Invalid Authentication Token"})
    
    path = path.lstrip("/")
    safe_path = os.path.normpath(os.path.join(STORAGE_ROOT, path))
    
    if not safe_path.startswith(STORAGE_ROOT):
        return json.dumps({"error": "Access Denied (Directory Traversal)"})
    
    if not os.path.exists(safe_path):
        return json.dumps({"error": f"File not found: {path}"})
    
    if os.path.isdir(safe_path):
        return json.dumps({"error": "Cannot delete directories"})
    
    try:
        os.remove(safe_path)
        return json.dumps({"success": True, "deleted": path})
    except Exception as e:
        return json.dumps({"error": f"Error deleting file: {str(e)}"})


if __name__ == "__main__":
    mcp.run()

