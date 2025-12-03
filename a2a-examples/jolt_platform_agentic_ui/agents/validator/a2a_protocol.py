from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    ERROR_REPORT = "error_report"
    PATCH_PROPOSAL = "patch_proposal"
    VERIFICATION_REQUEST = "verification_request"
    VERIFICATION_RESULT = "verification_result"
    DIAGNOSTIC_QUERY = "diagnostic_query"
    DIAGNOSTIC_RESPONSE = "diagnostic_response"

class A2AMessage(BaseModel):
    """Agent-to-Agent communication message"""
    message_type: MessageType
    sender: str  # "generator" or "validator"
    receiver: str
    content: Dict[str, Any]
    conversation_id: str

class ErrorReport(BaseModel):
    """Error report sent from Validator to Generator"""
    path: str  # JSONPath where error occurred
    expected: Any
    actual: Any
    error_description: str

class PatchProposal(BaseModel):
    """Patch proposal sent from Generator to Validator"""
    updated_spec: Dict[str, Any]
    changes_description: str
    iteration: int

class VerificationResult(BaseModel):
    """Validation result sent from Validator to Generator"""
    is_valid: bool
    errors: List[ErrorReport] = []
    success_message: Optional[str] = None

class A2AProtocol:
    """Manages Agent-to-Agent communication protocol"""
    
    def __init__(self):
        self.conversation_history: List[A2AMessage] = []
        self.max_iterations = 3
    
    def send_error_report(self, errors: List[ErrorReport], conversation_id: str) -> A2AMessage:
        """Validator sends error report to Generator"""
        logger.info(f"A2A: Sending ERROR_REPORT for conversation {conversation_id}. Error count: {len(errors)}")
        message = A2AMessage(
            message_type=MessageType.ERROR_REPORT,
            sender="validator",
            receiver="generator",
            content={"errors": [e.model_dump() for e in errors]},
            conversation_id=conversation_id
        )
        self.conversation_history.append(message)
        logger.debug(f"A2A Message Content: {json.dumps(message.model_dump(), default=str)}")
        return message
    
    def send_patch_proposal(self, patch: PatchProposal, conversation_id: str) -> A2AMessage:
        """Generator sends patch proposal to Validator"""
        logger.info(f"A2A: Sending PATCH_PROPOSAL for conversation {conversation_id}. Iteration: {patch.iteration}")
        message = A2AMessage(
            message_type=MessageType.PATCH_PROPOSAL,
            sender="generator",
            receiver="validator",
            content=patch.model_dump(),
            conversation_id=conversation_id
        )
        self.conversation_history.append(message)
        logger.debug(f"A2A Message Content: {json.dumps(message.model_dump(), default=str)}")
        return message
    
    def send_verification_result(self, result: VerificationResult, conversation_id: str) -> A2AMessage:
        """Validator sends verification result to Generator"""
        logger.info(f"A2A: Sending VERIFICATION_RESULT for conversation {conversation_id}. Valid: {result.is_valid}")
        message = A2AMessage(
            message_type=MessageType.VERIFICATION_RESULT,
            sender="validator",
            receiver="generator",
            content=result.model_dump(),
            conversation_id=conversation_id
        )
        self.conversation_history.append(message)
        logger.debug(f"A2A Message Content: {json.dumps(message.model_dump(), default=str)}")
        return message
    
    def get_conversation_history(self) -> List[A2AMessage]:
        """Get full conversation history"""
        return self.conversation_history
