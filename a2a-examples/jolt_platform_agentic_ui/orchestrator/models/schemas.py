from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class TaskType(str, Enum):
    GENERATE_SPEC = "generate_spec"
    VALIDATE_SPEC = "validate_spec"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

class AuthMapping(BaseModel):
    user_id: str
    gdrive_token: str
    target_directory_id: str

class TaskRequest(BaseModel):
    task_type: TaskType
    input_file_path: str
    output_file_path: str
    auth_mapping: AuthMapping
    description: Optional[str] = None

class ValidationReport(BaseModel):
    is_valid: bool
    errors: List[str] = []
    diff_summary: Optional[str] = None
    generated_spec: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
