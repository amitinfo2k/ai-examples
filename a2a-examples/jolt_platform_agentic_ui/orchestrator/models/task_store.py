"""
Task store for tracking workflow progress
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
import threading


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    VALIDATING = "validating"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class WorkflowProgress(BaseModel):
    task_id: str
    status: WorkflowStatus
    current_step: str
    logs: List[str]
    progress_percentage: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TaskStore:
    """Thread-safe in-memory task store"""
    
    def __init__(self):
        self._tasks: Dict[str, WorkflowProgress] = {}
        self._lock = threading.Lock()
    
    def create_task(self, task_id: str) -> WorkflowProgress:
        """Create a new task"""
        with self._lock:
            now = datetime.now()
            task = WorkflowProgress(
                task_id=task_id,
                status=WorkflowStatus.PENDING,
                current_step="Initializing workflow",
                logs=["Workflow created"],
                progress_percentage=0,
                created_at=now,
                updated_at=now
            )
            self._tasks[task_id] = task
            return task
    
    def update_task(
        self,
        task_id: str,
        status: Optional[WorkflowStatus] = None,
        current_step: Optional[str] = None,
        log: Optional[str] = None,
        logs: Optional[List[str]] = None,
        progress_percentage: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> WorkflowProgress:
        """Update task progress"""
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            
            if status is not None:
                task.status = status
            if current_step is not None:
                task.current_step = current_step
            if log is not None:
                task.logs.append(log)
            if logs is not None:
                task.logs.extend(logs)
            if progress_percentage is not None:
                task.progress_percentage = min(100, progress_percentage)
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            
            task.updated_at = datetime.now()
            return task
    
    def get_task(self, task_id: str) -> Optional[WorkflowProgress]:
        """Get task by ID"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def delete_task(self, task_id: str):
        """Delete a task"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]


# Global task store instance
task_store = TaskStore()
