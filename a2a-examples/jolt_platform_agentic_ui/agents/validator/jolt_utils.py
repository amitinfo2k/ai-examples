"""
Jolt transformation utility using MCP server's SSE endpoint.
"""
import asyncio
import json
# Import Client from the fastmcp.client module
from fastmcp.client import Client, SSETransport
from typing import Dict, Any, List, Union

import os

# Get MCP Server URL from environment or default to localhost
base_url = os.getenv("MCP_SERVICE_URL", "http://localhost:8081")
if not base_url.endswith("/sse"):
    base_url += "/sse"
MCP_JOLT_SERVER_URL = base_url

async def apply_jolt_shift_async(input_data: Union[Dict[str, Any], str], spec: Union[List[Dict[str, Any]], str]) -> Dict[str, Any]:
    """
    Apply Jolt transformation using the MCP server's transform tool.
    
    Args:
        input_data: The input JSON data (either as a dict or JSON string)
        spec: The Jolt specification (either as a list of dicts or JSON string)
        
    Returns:
        The transformed JSON data as a dictionary
    """
    try:
        print("\n=== Initializing MCP Client ===")
        print(f"Connecting to MCP Server at: {MCP_JOLT_SERVER_URL}")
        transport = SSETransport(MCP_JOLT_SERVER_URL)
        async with Client(transport=transport) as jolt_client:
            # Convert input data and spec to JSON strings if they aren't already
            input_json_str = json.dumps(input_data) if isinstance(input_data, (dict, list)) else input_data
            jolt_spec_str = json.dumps(spec) if isinstance(spec, (dict, list)) else spec
            
            print(f"\n=== Calling transform tool ===")
            print("Input JSON type:", type(input_json_str))
            print("JOLT Spec type:", type(jolt_spec_str))
            
            # Debug: Print first 100 chars of each to avoid huge logs
            debug_input = str(input_json_str)[:100] + ('...' if len(str(input_json_str)) > 100 else '')
            debug_spec = str(jolt_spec_str)[:100] + ('...' if len(str(jolt_spec_str)) > 100 else '')
            print(f"Input JSON (first 100 chars): {debug_input}")
            print(f"JOLT Spec (first 100 chars): {debug_spec}")
            
            # Call the transform tool
            result = await jolt_client.call_tool(
                "transform",
                {
                    "input_json": input_json_str,
                    "jolt_spec": jolt_spec_str
                }
            )
            
            print("\n=== Transformation Result ===")
            print("Result type:", type(result))
            
            # Handle CallToolResult object from MCP server
            if hasattr(result, 'content') and hasattr(result, 'is_error'):
                # This is a CallToolResult object
                print("Detected CallToolResult object")
                if result.is_error:
                    error_text = result.content[0].text if result.content else "Unknown error"
                    raise Exception(f"MCP tool returned error: {error_text}")
                
                # Extract the text content from the first content item
                if result.content and len(result.content) > 0:
                    text_content = result.content[0].text
                    print(f"Extracted text content: {text_content[:200]}...")
                    try:
                        result = json.loads(text_content)
                        print("Successfully parsed text content as JSON")
                    except json.JSONDecodeError as e:
                        print(f"Could not parse text content as JSON: {str(e)}")
                        raise Exception(f"Invalid JSON from MCP server: {text_content}")
                else:
                    raise Exception("CallToolResult has no content")
            
            # The result might be a string that needs to be parsed as JSON
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                    print("Parsed result as JSON")
                except json.JSONDecodeError as e:
                    print(f"Could not parse result as JSON: {str(e)}")
            
            # Handle different possible response formats
            if isinstance(result, dict):
                if "data" in result:
                    return result["data"]
                elif "result" in result:
                    return result["result"]
                return result
            elif isinstance(result, (list, str, int, float, bool)) or result is None:
                return {"result": result}
            else:
                return {"result": str(result)}

    except Exception as e:
        error_msg = f"Jolt transformation failed: {str(e)}"
        print(f"\n=== CRITICAL ERROR ===\n{error_msg}\n====================")
        raise Exception(error_msg)

def apply_jolt_shift(input_data: Dict[str, Any], spec: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synchronous wrapper for the async Jolt transformation.
    
    Args:
        input_data: The input JSON data
        spec: The Jolt specification
        
    Returns:
        The transformed JSON data
    """
    return asyncio.run(apply_jolt_shift_async(input_data, spec))

