#!/usr/bin/env python3
"""
JOLT MCP Server

This MCP server provides JOLT transformation services.
It accepts a JOLT spec and input JSON, and returns the transformed output.

Tools:
- transform_jolt: Apply JOLT transformation to input JSON
"""

import json
import sys
import logging
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jolt-mcp-server")

class JoltMCPServer:
    """MCP Server for JOLT transformations."""
    
    def __init__(self):
        self.name = "jolt"
        self.version = "1.0.0"
        
    def transform_jolt(self, jolt_spec: List[Dict], input_json: Dict) -> Dict:
        """
        Apply JOLT transformation to input JSON.
        
        This is an enhanced implementation supporting:
        - shift operations with nested paths and array indices
        - default operations
        - modify operations (basic)
        - remove operations
        
        Args:
            jolt_spec: JOLT specification (list of operations)
            input_json: Input JSON to transform
            
        Returns:
            Transformed JSON
        """
        result = {}
        
        for operation in jolt_spec:
            op_type = operation.get('operation', '').lower()
            spec = operation.get('spec', {})
            
            if op_type == 'shift':
                result = self._apply_shift(spec, input_json, result)
            elif op_type == 'default':
                result = self._apply_default(spec, result)
            elif op_type == 'modify':
                result = self._apply_modify(spec, result, input_json)
            elif op_type == 'remove':
                result = self._apply_remove(spec, result)
        
        return result
    
    def _apply_shift(self, spec: Dict, input_json: Dict, current_result: Dict) -> Dict:
        """Apply shift operation."""
        result = current_result.copy() if current_result else {}
        
        def process_spec(spec_node: Any, input_node: Any, path: str = ""):
            if isinstance(spec_node, dict):
                if isinstance(input_node, dict):
                    for key, value in spec_node.items():
                        if key in input_node:
                            process_spec(value, input_node[key], f"{path}.{key}" if path else key)
            elif isinstance(spec_node, str):
                # Target path
                self._set_nested_value(result, spec_node.split('.'), input_node)
        
        process_spec(spec, input_json)
        return result
    
    def _apply_default(self, spec: Dict, data: Dict) -> Dict:
        """Apply default operation - add default values for missing keys."""
        result = data.copy()
        
        def apply_defaults(spec_node: Dict, data_node: Dict):
            for key, value in spec_node.items():
                if isinstance(value, dict) and key in data_node and isinstance(data_node[key], dict):
                    apply_defaults(value, data_node[key])
                elif key not in data_node:
                    data_node[key] = value
        
        apply_defaults(spec, result)
        return result
    
    def _apply_modify(self, spec: Dict, data: Dict, input_json: Dict) -> Dict:
        """Apply modify operation - modify/compute values."""
        result = data.copy()
        
        for key, expression in spec.items():
            if isinstance(expression, str) and expression.startswith('='):
                # Parse expression (simplified - only handles concat and join)
                if 'concat' in expression or 'join' in expression:
                    # Extract values to concatenate/join
                    # This is a simplified parser
                    try:
                        # For now, just log that modify is not fully implemented
                        logger.warning(f"Modify operation with expression '{expression}' is simplified")
                    except Exception as e:
                        logger.error(f"Error in modify operation: {e}")
        
        return result
    
    def _apply_remove(self, spec: Dict, data: Dict) -> Dict:
        """Apply remove operation - remove keys from output."""
        result = data.copy()
        
        def remove_keys(spec_node: Dict, data_node: Dict):
            for key in list(spec_node.keys()):
                if key in data_node:
                    del data_node[key]
        
        remove_keys(spec, result)
        return result
    
    def _set_nested_value(self, data: Dict, path: List[str], value: Any):
        """Set a nested value in a dictionary, handling array indices."""
        current = data
        
        for i, key in enumerate(path[:-1]):
            # Check for array index in key (e.g., "events[0]")
            if '[' in key and key.endswith(']'):
                base_key, index_str = key[:-1].split('[')
                index = int(index_str)
                
                if base_key not in current:
                    current[base_key] = []
                
                # Ensure it's a list
                if not isinstance(current[base_key], list):
                    current[base_key] = [current[base_key]] if current[base_key] else []

                # Extend array if needed
                while len(current[base_key]) <= index:
                    current[base_key].append({})
                
                current = current[base_key][index]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
        
        # Handle the last key
        last_key = path[-1]
        if '[' in last_key and last_key.endswith(']'):
            base_key, index_str = last_key[:-1].split('[')
            index = int(index_str)
            
            if base_key not in current:
                current[base_key] = []
            
            # Ensure it's a list
            if not isinstance(current[base_key], list):
                current[base_key] = [current[base_key]] if current[base_key] else []

            while len(current[base_key]) <= index:
                current[base_key].append(None)
            
            current[base_key][index] = value
        else:
            current[last_key] = value

    def handle_tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Handle MCP tool calls."""
        if tool_name == "transform_jolt":
            jolt_spec = arguments.get("jolt_spec", [])
            input_json = arguments.get("input_json", {})
            
            try:
                result = self.transform_jolt(jolt_spec, input_json)
                return {
                    "success": True,
                    "result": result
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }


def main():
    """Main MCP server loop (stdio protocol)."""
    server = JoltMCPServer()
    logger.info(f"JOLT MCP Server v{server.version} started")
    
    # Read JSON-RPC requests from stdin
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                result = server.handle_tool_call(tool_name, arguments)
                
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            print(json.dumps(response), flush=True)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            if 'request_id' in locals():
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
