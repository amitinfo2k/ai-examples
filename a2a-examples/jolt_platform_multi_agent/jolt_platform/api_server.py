"""
FastAPI Server for JOLT Platform
Provides REST API endpoints for JOLT specification creation and validation
using both CrewAI and LangChain agents.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import threading
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from jolt_platform.unified_platform import JoltPlatform
from jolt_platform.messaging import get_message_bus, Message

# Initialize FastAPI app
app = FastAPI(
    title="JOLT Multi-Agent Platform API",
    description="Multi-agent platform for JOLT specification creation (CrewAI) and validation (LangChain)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize platform
platform = JoltPlatform(output_dir="./api_output")
platform.load_default_agents()

# Store for async jobs
# Store for async jobs
jobs = {}

# Background consumer for A2A status updates
def start_status_consumer():
    """Starts a background thread to consume Kafka messages and update job status."""
    try:
        bus = get_message_bus()
        if "Kafka" not in type(bus).__name__:
            print("⚠️ Not using Kafka, background status updates disabled.")
            return

        def update_job_status(msg: Message):
            payload = msg.payload
            job_id = payload.get("job_id")
            if job_id and job_id in jobs:
                print(f"🔄 Updating status for job {job_id}: {msg.type}")
                if msg.type == "WORKFLOW_COMPLETE":
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["result"] = payload.get("result")
                    jobs[job_id]["completed_at"] = datetime.now().isoformat()
                elif msg.type == "WORKFLOW_ERROR":
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = payload.get("error")
                    jobs[job_id]["completed_at"] = datetime.now().isoformat()

        bus.subscribe("WORKFLOW_COMPLETE", update_job_status)
        bus.subscribe("WORKFLOW_ERROR", update_job_status)
        
        print("🚀 Starting background status consumer...")
        bus.start_consuming()
    except Exception as e:
        print(f"❌ Error in background consumer: {e}")

@app.on_event("startup")
async def startup_event():
    # Start consumer in a separate thread so it doesn't block API
    thread = threading.Thread(target=start_status_consumer, daemon=True)
    thread.start()


# Pydantic models for request/response
class JoltCreationRequest(BaseModel):
    """Request model for JOLT specification creation."""
    input_json: Dict[str, Any]
    expected_output: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_json": {
                    "user": {
                        "firstName": "John",
                        "lastName": "Doe"
                    }
                },
                "expected_output": {
                    "fullName": "John Doe"
                }
            }
        }


class JoltValidationRequest(BaseModel):
    """Request model for JOLT specification validation."""
    jolt_spec: Dict[str, Any]
    input_json: Dict[str, Any]
    expected_output: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "jolt_spec": [
                    {
                        "operation": "shift",
                        "spec": {
                            "user": {
                                "firstName": "fullName"
                            }
                        }
                    }
                ],
                "input_json": {
                    "user": {
                        "firstName": "John"
                    }
                },
                "expected_output": {
                    "fullName": "John"
                }
            }
        }


class JoltWorkflowRequest(BaseModel):
    """Request model for complete JOLT workflow."""
    input_json: Dict[str, Any]
    expected_output: Dict[str, Any]
    async_mode: Optional[bool] = False
    execution_mode: Optional[str] = "traditional"  # "traditional" or "a2a"


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "JOLT Platform API",
        "version": "2.0.0",
        "agents": {
            "creation": "CrewAI",
            "validation": "LangChain"
        },
        "execution_modes": {
            "traditional": "Platform orchestrates agents (default)",
            "a2a": "Event-driven messaging (Agent-to-Agent protocol)"
        },
        "endpoints": {
            "POST /create": "Create JOLT specification using CrewAI agent",
            "POST /validate": "Validate JOLT specification using LangChain agent",
            "POST /workflow": "Complete workflow (create + validate). Supports 'execution_mode' parameter",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "crewai": "active",
            "langchain": "active"
        }
    }


@app.post("/create")
async def create_jolt_spec(request: JoltCreationRequest):
    """
    Create JOLT specification using CrewAI agent.
    
    This endpoint uses the CrewAI agent to generate a JOLT transformation
    specification based on input and expected output JSON examples.
    """
    try:
        jolt_spec = platform.create_spec_only(
            request.input_json,
            request.expected_output
        )
        
        return {
            "status": "success",
            "agent": "CrewAI",
            "timestamp": datetime.now().isoformat(),
            "jolt_spec": jolt_spec
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating JOLT spec: {str(e)}")


@app.post("/validate")
async def validate_jolt_spec(request: JoltValidationRequest):
    """
    Validate JOLT specification using LangChain agent.
    
    This endpoint uses the LangChain agent to validate a JOLT specification
    by applying it to input JSON and comparing with expected output.
    """
    try:
        validation_report = platform.validate_spec_only(
            request.jolt_spec,
            request.input_json,
            request.expected_output
        )
        
        return {
            "status": "success",
            "agent": "LangChain",
            "timestamp": datetime.now().isoformat(),
            "validation_report": validation_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating JOLT spec: {str(e)}")


@app.post("/workflow")
async def complete_workflow(request: JoltWorkflowRequest, background_tasks: BackgroundTasks):
    """
    Complete JOLT workflow: Create specification with CrewAI and validate with LangChain.
    
    This endpoint orchestrates both agents:
    1. CrewAI agent creates the JOLT specification
    2. LangChain agent validates the specification
    
    Parameters:
    - execution_mode: "traditional" (platform orchestrates) or "a2a" (event-driven messaging)
    - async_mode: Run in background (True) or synchronously (False)
    """
    # Enable A2A mode if requested
    if request.execution_mode == "a2a" and not hasattr(platform, 'bus'):
        platform.enable_a2a_mode()
    
    if request.async_mode:
        # Async mode: Return job ID immediately
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "execution_mode": request.execution_mode
        }
        
        # Add task to background
        background_tasks.add_task(
            run_workflow_async,
            job_id,
            request.input_json,
            request.expected_output,
            request.execution_mode
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Workflow started in background",
            "execution_mode": request.execution_mode,
            "check_status_at": f"/workflow/{job_id}"
        }
    else:
        # Sync mode: Run workflow and return results
        try:
            if request.execution_mode == "a2a":
                # Use A2A messaging protocol
                # For A2A, we always generate a job_id so we can track it
                job_id = str(uuid.uuid4())
                
                # Register job in memory so status endpoint works
                jobs[job_id] = {
                    "status": "running",
                    "created_at": datetime.now().isoformat(),
                    "execution_mode": "a2a"
                }
                
                result = platform.run_a2a_workflow(
                    request.input_json,
                    request.expected_output,
                    job_id=job_id
                )
                # Return standardized response
                return {
                    "status": "success",
                    "execution_mode": "a2a",
                    "job_id": job_id,
                    "agents": {
                        "creation": "CrewAI",
                        "validation": "LangChain"
                    },
                    "timestamp": datetime.now().isoformat(),
                    "result": result
                }
            else:
                # Use traditional orchestration
                result = platform.create_and_validate(
                    request.input_json,
                    request.expected_output,
                    save_outputs=True
                )
                
                # Extract validation report
                validation_report = result.get("validation_report", {})
                
                return {
                    "status": "success",
                    "agent": "LangChain",
                    "timestamp": datetime.now().isoformat(),
                    "validation_report": validation_report
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")


@app.get("/workflow/{job_id}")
async def get_workflow_status(job_id: str):
    """Get status of an async workflow job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


def run_workflow_async(job_id: str, input_json: Dict[str, Any], expected_output: Dict[str, Any], execution_mode: str = "traditional"):
    """Background task for async workflow execution."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        if execution_mode == "a2a":
            result = platform.run_a2a_workflow(input_json, expected_output, job_id=job_id)
        else:
            result = platform.create_and_validate(input_json, expected_output, save_outputs=True)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
