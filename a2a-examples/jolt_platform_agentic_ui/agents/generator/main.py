from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Union
import logging

from agents.generator.crew_agent import JoltSpecGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jolt Generator Service", version="0.1.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    input_file_path: str
    output_file_path: str
    auth_token: str


class RefineRequest(BaseModel):
    current_spec: Union[List[Dict[str, Any]], Dict[str, Any]]
    error_report: List[Dict[str, Any]]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "generator"}


@app.post("/generate")
async def generate_jolt_spec(request: GenerateRequest):
    """Generate a Jolt specification from input and output files"""
    try:
        logger.info(f"Generating Jolt spec for input={request.input_file_path}, output={request.output_file_path}")
        generator = JoltSpecGenerator()
        spec = generator.generate(
            input_path=request.input_file_path,
            output_path=request.output_file_path,
            auth_token=request.auth_token
        )
        return {"jolt_spec": spec, "status": "success"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating Jolt spec: {error_msg}", exc_info=True)
        
        # Check if it's a quota error
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "API Quota Exceeded",
                    "message": "Gemini API quota exhausted. Please wait for quota reset or use a different API key.",
                    "suggestion": "Set GEMINI_MODEL environment variable to try a different model or upgrade to paid tier.",
                    "original_error": error_msg
                }
            )
        
        raise HTTPException(status_code=500, detail=f"Generation failed: {error_msg}")


@app.post("/refine")
async def refine_jolt_spec(request: RefineRequest):
    """Refine a Jolt specification based on validation errors"""
    try:
        logger.info(f"Refining Jolt spec based on {len(request.error_report)} errors")
        generator = JoltSpecGenerator()
        refined_spec = generator.refine_spec(
            current_spec=request.current_spec,
            error_report=request.error_report
        )
        return {"jolt_spec": refined_spec, "status": "success"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error refining Jolt spec: {error_msg}", exc_info=True)
        
        # Check if it's a quota error
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "API Quota Exceeded",
                    "message": "Gemini API quota exhausted. Please wait for quota reset or use a different API key.",
                    "suggestion": "Set GEMINI_MODEL environment variable to try a different model or upgrade to paid tier.",
                    "original_error": error_msg
                }
            )
        
        raise HTTPException(status_code=500, detail=f"Refinement failed: {error_msg}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
