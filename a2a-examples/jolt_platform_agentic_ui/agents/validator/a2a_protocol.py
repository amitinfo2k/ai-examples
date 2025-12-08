from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"

class ADKTask(BaseModel):
    """Google ADK Task Object"""
    id: str
    status: TaskStatus
    type: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ErrorReport(BaseModel):
    """Error report sent from Validator to Generator"""
    path: str  # JSONPath where error occurred
    expected: Any
    actual: Any
    error_description: str

class VerificationResult(BaseModel):
    """Validation result sent from Validator to Generator"""
    is_valid: bool
    errors: List[ErrorReport] = []
    success_message: Optional[str] = None

class A2AProtocol:
    """Manages Agent-to-Agent communication protocol (ADK Compliant)"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
    
    def log_interaction(self, source: str, target: str, action: str, details: Any):
        """Log an interaction between agents"""
        entry = {
            "source": source,
            "target": target,
            "action": action,
            "details": details,
            "timestamp": "iso-timestamp-here" # In a real app, use datetime.now().isoformat()
        }
        self.conversation_history.append(entry)
        logger.info(f"A2A Interaction: {source} -> {target} [{action}]")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        return self.conversation_history
