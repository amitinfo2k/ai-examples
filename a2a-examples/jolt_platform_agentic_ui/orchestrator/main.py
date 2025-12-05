from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set
from orchestrator.models.schemas import TaskRequest, TaskResponse, TaskStatus
from orchestrator.models.task_store import task_store, WorkflowStatus
from typing import Union, List, Any, Optional
from pydantic import BaseModel
import uuid
import logging
import traceback
import asyncio
import httpx
import os
import json

class ValidateRequestBody(BaseModel):
    jolt_spec: Union[List[Dict[str, Any]], Dict[str, Any]]
    expected_output: Optional[Dict[str, Any]] = None


class PromptRefineRequestBody(BaseModel):
    current_spec: Union[List[Dict[str, Any]], Dict[str, Any]]
    user_feedback: str
    input_json: Optional[Dict[str, Any]] = None
    expected_output: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[Dict[str, Any]]] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Jolt Orchestrator", version="0.1.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
GENERATOR_URL = os.getenv("GENERATOR_URL", "http://localhost:8081")
VALIDATOR_URL = os.getenv("VALIDATOR_URL", "http://localhost:8080")

logger.info(f"Generator Service URL: {GENERATOR_URL}")
logger.info(f"Validator Service URL: {VALIDATOR_URL}")

# In-memory storage for tasks (replace with DB later)
tasks = {}

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()
        self.active_connections[task_id].add(websocket)

    def disconnect(self, task_id: str, websocket: WebSocket):
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def broadcast(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            for connection in list(self.active_connections[task_id]):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error broadcasting to WebSocket: {e}")
                    self.disconnect(task_id, connection)

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "orchestrator"}


# ===== GDrive File Management Endpoints =====

@app.get("/files/list")
async def list_files(folder: str = "", auth_token: str = "valid_token"):
    """List files in the mock GDrive storage"""
    try:
        from orchestrator.core.mcp_client import mcp_client
        result = await mcp_client.list_files(auth_token, folder)
        return result
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/read")
async def read_file(path: str, auth_token: str = "valid_token"):
    """Read a file from mock GDrive storage"""
    try:
        from orchestrator.core.mcp_client import mcp_client
        content = await mcp_client.read_file(path, auth_token)
        if content.startswith("Error:"):
            raise HTTPException(status_code=400, detail=content)
        return {"path": path, "content": content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class WriteFileRequest(BaseModel):
    path: str
    content: str
    auth_token: str = "valid_token"


@app.post("/files/write")
async def write_file(request: WriteFileRequest):
    """Write/upload a file to mock GDrive storage"""
    try:
        from orchestrator.core.mcp_client import mcp_client
        result = await mcp_client.write_file(request.path, request.content, request.auth_token)
        return result
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/delete")
async def delete_file(path: str, auth_token: str = "valid_token"):
    """Delete a file from mock GDrive storage"""
    try:
        from orchestrator.core.mcp_client import mcp_client
        result = await mcp_client.delete_file(path, auth_token)
        return result
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    task_id = str(uuid.uuid4())
    task = TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task created successfully"
    )
    tasks[task_id] = {
        "request": request,
        "response": task
    }
    # TODO: Trigger Agent 1 (CrewAI) asynchronously
    return task

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]["response"]

@app.get("/test-mcp")
async def test_mcp(path: str = "input.json", token: str = "valid_token"):
    from orchestrator.core.mcp_client import mcp_client
    content = await mcp_client.read_file(path, token)
    return {"path": path, "content": content}

@app.post("/generate")
async def generate_jolt_spec(request: TaskRequest):
    """Generate a Jolt spec using CrewAI agent via HTTP"""
    
    # Update task status
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "request": request,
        "response": TaskResponse(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            message="Generation started"
        )
    }
    
    try:
        logger.info(f"Calling Generator service at {GENERATOR_URL}/generate")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GENERATOR_URL}/generate",
                json={
                    "input_file_path": request.input_file_path,
                    "output_file_path": request.output_file_path,
                    "auth_token": request.auth_mapping.gdrive_token
                }
            )
            response.raise_for_status()
            result = response.json()
        
        # Update task with result
        tasks[task_id]["response"] = TaskResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            message="Jolt spec generated successfully"
        )
        
        return tasks[task_id]["response"]
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling generator service: {str(e)}")
        tasks[task_id]["response"] = TaskResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            message=f"Generation failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error calling generator service: {str(e)}")
        tasks[task_id]["response"] = TaskResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            message=f"Generation failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate_jolt_spec(
    input_path: str,
    output_path: str,
    body: ValidateRequestBody,
    auth_token: str = "valid_token"
):
    """Validate a Jolt spec using LangGraph agent via HTTP"""
    from orchestrator.core.mcp_client import mcp_client
    
    try:
        # Read input file via MCP
        input_content = await mcp_client.read_file(input_path, auth_token)
        input_json = json.loads(input_content)
        
        # Determine expected output
        if body.expected_output:
            expected_output = body.expected_output
        else:
            # Read from file if not provided in body
            output_content = await mcp_client.read_file(output_path, auth_token)
            expected_output = json.loads(output_content)
        
        # Call validator service via HTTP
        logger.info(f"Calling Validator service at {VALIDATOR_URL}/validate")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{VALIDATOR_URL}/validate",
                json={
                    "input_json": input_json,
                    "expected_output": expected_output,
                    "jolt_spec": body.jolt_spec
                }
            )
            response.raise_for_status()
            result = response.json()
        
        return result
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling validator service: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Validator Service Error",
                "message": f"Failed to call validator service: {str(e)}",
                "traceback": traceback.format_exc().splitlines()
            }
        )
    except Exception as e:
        logger.error(f"Error during validation: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal Server Error",
                "message": f"An unexpected error occurred: {str(e)}",
                "traceback": traceback.format_exc().splitlines()
            }
        )


@app.post("/refine-with-prompt")
async def refine_jolt_spec_with_prompt(body: PromptRefineRequestBody):
    """Refine a Jolt specification based on user's natural language feedback"""
    try:
        logger.info(f"Forwarding prompt-based refinement request to Generator")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GENERATOR_URL}/refine-with-prompt",
                json={
                    "current_spec": body.current_spec,
                    "user_feedback": body.user_feedback,
                    "input_json": body.input_json,
                    "expected_output": body.expected_output,
                    "validation_errors": body.validation_errors
                }
            )
            response.raise_for_status()
            result = response.json()
        
        return result
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling generator service: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Generator Service Error",
                "message": f"Failed to call generator service: {str(e)}",
                "traceback": traceback.format_exc().splitlines()
            }
        )
    except Exception as e:
        logger.error(f"Error during prompt-based refinement: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal Server Error",
                "message": f"An unexpected error occurred: {str(e)}",
                "traceback": traceback.format_exc().splitlines()
            }
        )

async def _broadcast_task_update(task_id: str):
    """Helper function to get task and broadcast its status."""
    try:
        task = task_store.get_task(task_id)
        if task:
            payload = {
                "type": "status_update",
                "task_id": task_id,
                "status": task.status,
                "current_step": task.current_step,
                "progress_percentage": task.progress_percentage,
                "logs": task.logs[-15:],
                "timestamp": task.updated_at.isoformat(),
                "result": task.result  # Include the result field
            }
            await manager.broadcast(task_id, payload)
    except Exception as e:
        logger.error(f"Error broadcasting task update for {task_id}: {e}")

async def _run_workflow_background(task_id: str, request: TaskRequest):
    """Background task to run the workflow with progress tracking"""
    from orchestrator.core.mcp_client import mcp_client
    
    try:
        # Step 1: Generate Jolt Spec
        task_store.update_task(
            task_id,
            status=WorkflowStatus.GENERATING,
            current_step="Generating Jolt Specification",
            log="Starting Generation Phase...",
            progress_percentage=10
        )
        await _broadcast_task_update(task_id)
        
        logger.info(f"Calling Generator service at {GENERATOR_URL}/generate")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GENERATOR_URL}/generate",
                json={
                    "input_file_path": request.input_file_path,
                    "output_file_path": request.output_file_path,
                    "auth_token": request.auth_mapping.gdrive_token
                }
            )
            response.raise_for_status()
            generator_result = response.json()
            spec = generator_result.get("jolt_spec")
        
        task_store.update_task(
            task_id,
            log=f"Generated Jolt Spec with {len(spec) if isinstance(spec, list) else 1} operation(s)",
            progress_percentage=30
        )
        await _broadcast_task_update(task_id)
        
        # Step 2: Read input and output files
        task_store.update_task(
            task_id,
            current_step="Reading input and output files",
            log="Reading files for validation..."
        )
        await _broadcast_task_update(task_id)
        
        input_content = await mcp_client.read_file(
            request.input_file_path,
            request.auth_mapping.gdrive_token
        )
        output_content = await mcp_client.read_file(
            request.output_file_path,
            request.auth_mapping.gdrive_token
        )
        
        input_json = json.loads(input_content)
        expected_output = json.loads(output_content)
        
        # Step 3: A2A Validation with Collaborative Debugging
        # The validator will handle the refinement loop via A2A communication with generator
        task_store.update_task(
            task_id,
            status=WorkflowStatus.VALIDATING,
            current_step="Validating Jolt Specification with A2A Collaboration",
            log="Starting A2A Validation Phase (Validator will communicate directly with Generator if needed)...",
            progress_percentage=40
        )
        await _broadcast_task_update(task_id)
        
        max_retries = 3
        logger.info(f"Calling Validator service at {VALIDATOR_URL}/validate-with-a2a")
        async with httpx.AsyncClient(timeout=600.0) as client:  # Longer timeout for A2A loop
            response = await client.post(
                f"{VALIDATOR_URL}/validate-with-a2a",
                params={"max_retries": max_retries},
                json={
                    "input_json": input_json,
                    "expected_output": expected_output,
                    "jolt_spec": spec
                }
            )
            response.raise_for_status()
            validation_result = response.json()
        
        # Process validation result
        is_valid = validation_result.get('is_valid', False)
        final_spec = validation_result.get('jolt_spec', spec)
        attempts = validation_result.get('attempts', 1)
        
        # Add validation logs to task
        for log_msg in validation_result.get('logs', []):
            if log_msg:
                task_store.update_task(task_id, log=str(log_msg))
        
        task_store.update_task(
            task_id,
            log=f"A2A Validation completed after {attempts} attempt(s): {'✅ PASSED' if is_valid else '❌ FAILED'}",
            progress_percentage=90
        )
        await _broadcast_task_update(task_id)
        
        # Prepare final result
        final_status = WorkflowStatus.COMPLETED if is_valid else WorkflowStatus.NEEDS_REVIEW
        
        # Ensure validation result is JSON serializable before storing
        serializable_validation_result = validation_result.copy()
        if 'errors' in serializable_validation_result:
            serializable_validation_result['errors'] = [
                err.dict() if hasattr(err, 'dict') else err
                for err in serializable_validation_result.get('errors', [])
            ]

        result_data = {
            "jolt_spec": final_spec,
            "input_json": input_json,
            "expected_output": expected_output,
            "input_file_path": request.input_file_path,
            "output_file_path": request.output_file_path,
            "validation": serializable_validation_result
        }
        
        task_store.update_task(
            task_id,
            status=final_status,
            current_step="Workflow completed",
            progress_percentage=100,
            result=result_data
        )
        
        # Send final update to WebSocket clients
        # Final broadcast is handled by the main update_task call below
        await _broadcast_task_update(task_id)
        
    except Exception as e:
        logger.error(f"Error during workflow: {e}\n{traceback.format_exc()}")
        task_store.update_task(
            task_id,
            status=WorkflowStatus.FAILED,
            current_step="Workflow failed",
            error=str(e),
            log=f"Error: {str(e)}",
            progress_percentage=100
        )


@app.post("/workflow/generate-and-validate")
async def complete_workflow(request: TaskRequest, background_tasks: BackgroundTasks):
    """Complete workflow: Generate spec then validate it (runs in background)"""
    task_id = str(uuid.uuid4())
    
    # Create task in store
    task_store.create_task(task_id)
    
    # Run workflow in background
    background_tasks.add_task(_run_workflow_background, task_id, request)
    
    # Return task ID immediately
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Workflow started in background. Use /workflow/status/{task_id} to check progress."
    }


@app.websocket("/ws/status/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time status updates"""
    task = task_store.get_task(task_id)
    if not task:
        await websocket.close(code=1008, reason="Task not found")
        return
    
    await manager.connect(task_id, websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status_update",
            "task_id": task_id,
            "status": task.status,
            "current_step": task.current_step,
            "logs": task.logs[-10:],
            "progress_percentage": task.progress_percentage,
            "result": task.result if hasattr(task, 'result') else None,
            "error": task.error if hasattr(task, 'error') else None
        })
        
        # Keep connection open
        while True:
            # Client can send pings to keep the connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(task_id, websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(task_id, websocket)

@app.get("/workflow/status/{task_id}")
async def get_workflow_status(task_id: str):
    """Get the status of a running or completed workflow (HTTP fallback)"""
    task = task_store.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    response = {
        "task_id": task.task_id,
        "status": task.status,
        "current_step": task.current_step,
        "logs": task.logs,
        "progress_percentage": task.progress_percentage,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat()
    }
    
    # Include result if completed
    if task.result:
        response["result"] = task.result
    
    # Include error if failed
    if task.error:
        response["error"] = task.error
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
