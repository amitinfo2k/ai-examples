from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Union, List, Optional
import logging

from agents.validator.langgraph_agent import JoltValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jolt Validator Service", version="0.1.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValidateRequest(BaseModel):
    input_json: Dict[str, Any]
    expected_output: Dict[str, Any]
    jolt_spec: Union[List[Dict[str, Any]], Dict[str, Any]]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "validator"}


@app.post("/validate")
async def validate_jolt_spec(request: ValidateRequest):
    """Validate a Jolt specification by transforming input and comparing with expected output (single attempt)"""
    try:
        logger.info(f"Validating Jolt spec")
        validator = JoltValidator()
        result = await validator.validate_spec(
            input_json=request.input_json,
            expected_output=request.expected_output,
            jolt_spec=request.jolt_spec
        )
        return result
    except Exception as e:
        logger.error(f"Error validating Jolt spec: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/validate-with-a2a")
async def validate_with_a2a_collaboration(request: ValidateRequest, max_retries: int = 3):
    """
    Validate with A2A collaborative debugging - validator will automatically 
    communicate with generator to refine the spec if validation fails
    """
    try:
        logger.info(f"Validating Jolt spec with A2A collaboration (max_retries={max_retries})")
        validator = JoltValidator()
        result = await validator.validate_with_retries(
            input_json=request.input_json,
            expected_output=request.expected_output,
            jolt_spec=request.jolt_spec,
            max_retries=max_retries
        )
        return result
    except Exception as e:
        logger.error(f"Error during A2A validation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A2A validation failed: {str(e)}")


from fastapi.responses import FileResponse

@app.get("/.well-known/agent.json")
async def get_agent_card():
    """Serve the Agent Card for discovery"""
    return FileResponse("agents/validator/agent.json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
