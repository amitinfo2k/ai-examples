#!/usr/bin/env python3
"""
HTTP-based MCP Server for JOLT Transformations
Exposes the MCP JOLT server as a REST API for Kubernetes deployment.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import json

app = FastAPI(
    title="JOLT MCP Server",
    description="Model Context Protocol server for JOLT transformations",
    version="1.0.0"
)

class TransformRequest(BaseModel):
    """Request model for JOLT transformation."""
    jolt_spec: List[Dict[str, Any]]
    input_json: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "jolt_spec": [
                    {
                        "operation": "shift",
                        "spec": {
                            "user": {
                                "name": "fullName"
                            }
                        }
                    }
                ],
                "input_json": {
                    "user": {
                        "name": "Alice"
                    }
                }
            }
        }

class TransformResponse(BaseModel):
    """Response model for JOLT transformation."""
    success: bool
    result: Dict[str, Any] = None
    error: str = None


class JoltTransformer:
    """Custom JOLT transformation implementation."""
    
    def transform(self, jolt_spec: List[Dict], input_json: Dict) -> Dict:
        """Apply JOLT transformations."""
        result = {}
        
        for operation in jolt_spec:
            op_type = operation.get('operation', '')
            spec = operation.get('spec', {})
            
            if op_type == 'shift':
                result = self._apply_shift(spec, input_json)
            elif op_type == 'default':
                result = self._apply_default(spec, result)
        
        return result
    
    def _apply_shift(self, spec: Dict, input_json: Dict) -> Dict:
        """Apply shift operation."""
        result = {}
        
        for key, value in spec.items():
            if isinstance(value, dict):
                # Nested transformation
                if key in input_json and isinstance(input_json[key], dict):
                    nested_result = self._apply_shift(value, input_json[key])
                    result.update(nested_result)
            elif isinstance(value, str):
                # Simple mapping
                if key in input_json:
                    # Parse the target path
                    target_parts = value.split('.')
                    self._set_nested_value(result, target_parts, input_json[key])
        
        return result
    
    def _apply_default(self, spec: Dict, data: Dict) -> Dict:
        """Apply default operation."""
        result = data.copy()
        
        for key, value in spec.items():
            if key not in result:
                result[key] = value
        
        return result
    
    def _set_nested_value(self, data: Dict, path: List[str], value: Any):
        """Set a nested value in a dictionary."""
        current = data
        
        for i, key in enumerate(path[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[path[-1]] = value


transformer = JoltTransformer()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "jolt-mcp-server"}


@app.post("/transform", response_model=TransformResponse)
async def transform(request: TransformRequest):
    """
    Perform JOLT transformation.
    
    Args:
        request: TransformRequest with jolt_spec and input_json
        
    Returns:
        TransformResponse with the transformed JSON or error
    """
    import json
    
    try:
        print("\n" + "=" * 60)
        print("🔄 MCP Server: Received transformation request")
        print(f"📥 MCP Server: Input JSON: {json.dumps(request.input_json, indent=2)}")
        print(f"📋 MCP Server: JOLT Spec ({len(request.jolt_spec)} operations):")
        print(json.dumps(request.jolt_spec, indent=2))
        
        # Validate spec
        if not request.jolt_spec or not isinstance(request.jolt_spec, list):
            raise ValueError("jolt_spec must be a non-empty list of operations")
        
        # Perform transformation
        print("🔧 MCP Server: Applying JOLT transformation...")
        result = transformer.transform(request.jolt_spec, request.input_json)
        print(f"✅ MCP Server: Transformation successful!")
        print(f"📤 MCP Server: Result: {json.dumps(result, indent=2)}")
        print("=" * 60 + "\n")
        
        return TransformResponse(
            success=True,
            result=result
        )
        
    except Exception as e:
        print(f"❌ MCP Server: Transformation failed: {str(e)}")
        print("=" * 60 + "\n")
        return TransformResponse(
            success=False,
            error=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "JOLT MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transform": "/transform (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
