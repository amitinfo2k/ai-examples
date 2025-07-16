"""
Log Analysis Agent implementation.

This agent analyzes logs from Kubernetes pods to identify issues
that might be causing CrashLoopBackOff states.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator

import google.generativeai as genai
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import TextPart
from a2a.types import Task as TaskRequest
from common.types import TaskResult
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from common.base_agent import BaseAgent
from common.agent_executor import GenericAgentExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")


class LogAnalysisRequest(BaseModel):
    """Request model for log analysis."""
    logs: str
    pod_name: str


class LogAnalysisAgent(BaseAgent):
    """Implementation of the Log Analysis Agent."""
    
    agent_name: str = Field(default="Log Analysis Agent")
    description: str = Field(
        default="Analyzes Kubernetes pod logs to identify issues that might be causing CrashLoopBackOff states"
    )
    content_types: list[str] = Field(default=["text/plain"])
    
    async def stream(self, query: str, context_id: str, task_id: str) -> AsyncGenerator[TextPart, None]:
        """Stream responses from the agent."""
        yield TextPart(text=f"Received query: {query}\nProcessing log analysis request...")
        
        try:
            # Simple query parsing - this could be improved with NLP or a more sophisticated parser
            query_lower = query.lower()
            
            if "analyze logs" in query_lower or "analyze pod logs" in query_lower:
                # Try to extract pod name and logs from the query
                # In a real system, we'd likely get the logs from the Kubernetes API
                # For now, we'll assume the logs are provided in the query after "logs:" or similar
                
                pod_name = None
                logs = None
                
                # Extract pod name
                if "pod:" in query_lower or "pod name:" in query_lower:
                    pod_start_idx = query_lower.find("pod:")
                    if pod_start_idx == -1:
                        pod_start_idx = query_lower.find("pod name:")
                        if pod_start_idx != -1:
                            pod_start_idx += 9  # length of "pod name:"
                    else:
                        pod_start_idx += 4  # length of "pod:"
                    
                    if pod_start_idx != -1:
                        pod_end_idx = query_lower.find("\n", pod_start_idx)
                        if pod_end_idx == -1:
                            pod_end_idx = len(query_lower)
                        pod_name = query[pod_start_idx:pod_end_idx].strip()
                
                # Extract logs
                if "logs:" in query_lower:
                    logs_start_idx = query.find("logs:")
                    if logs_start_idx != -1:
                        logs_start_idx += 5  # length of "logs:"
                        logs = query[logs_start_idx:].strip()
                
                # If we have both pod name and logs, analyze them
                if pod_name and logs:
                    yield TextPart(text=f"Analyzing logs for pod {pod_name}...")
                    try:
                        analysis = await self.analyze_logs(logs, pod_name)
                        yield TextPart(text=f"Log analysis: {analysis}")
                    except Exception as e:
                        yield TextPart(text=f"Error analyzing logs: {str(e)}")
                else:
                    # If we're missing information, ask for it
                    missing_info = []
                    if not pod_name:
                        missing_info.append("pod name")
                    if not logs:
                        missing_info.append("logs")
                    
                    yield TextPart(text=f"Missing information: {', '.join(missing_info)}. Please provide the complete information.")
                    yield TextPart(text="Format your request as:\nPod: <pod_name>\nLogs: <logs>")
            else:
                yield TextPart(text="I'm not sure what you're asking for. You can ask me to analyze logs for a specific pod.")
                yield TextPart(text="Format your request as:\nPod: <pod_name>\nLogs: <logs>")
        except Exception as e:
            yield TextPart(text=f"Error processing query: {str(e)}")
            import traceback
            yield TextPart(text=f"Traceback: {traceback.format_exc()}")
        
        yield TextPart(text="Query processing complete.")
        

    async def analyze_logs(self, logs: str, pod_name: str) -> str:
        """Analyze Kubernetes pod logs to identify issues."""
        try:
            # Check if logs are empty
            if not logs or logs.strip() == "":
                return "No logs available for analysis. The pod might not have started yet or logs might not be accessible."
            
            # Use Gemini to analyze the logs
            if not GEMINI_API_KEY:
                return "GEMINI_API_KEY not configured. Cannot perform log analysis."
            
            # Configure the model
            model = genai.GenerativeModel('gemini-pro')
            
            # Create a prompt for log analysis
            prompt = f"""
            You are a Kubernetes expert analyzing logs from a pod named '{pod_name}' that is experiencing issues.
            The pod might be in a CrashLoopBackOff state or having other problems.
            
            Please analyze these logs and identify:
            1. The root cause of any errors or crashes
            2. Potential solutions to fix the issues
            3. Any missing dependencies, configuration issues, or resource constraints
            
            Here are the logs to analyze:
            ```
            {logs}
            ```
            
            Provide a clear, concise analysis with specific recommendations for fixing the issues.
            If you can't determine the cause from these logs, suggest what additional information would be helpful.
            """
            
            # Generate the analysis
            response = await model.generate_content_async(prompt)
            analysis = response.text
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            return f"Failed to analyze logs: {str(e)}"


class LogAnalysisAgentExecutor(GenericAgentExecutor):
    """Executor for the Log Analysis Agent."""

    def __init__(self):
        """Initialize the Log Analysis Agent Executor."""
        agent = LogAnalysisAgent(
            agent_name="Log Analysis Agent",
            description="Agent for analyzing Kubernetes pod logs",
            content_types=["text/plain"]
        )
        super().__init__(agent)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the agent execution."""
        await event_queue.enqueue_event(new_agent_text_message("Log Analysis Agent task cancelled"))
    
    async def analyze_logs(self, request: TaskRequest) -> TaskResult:
        """
        Analyze Kubernetes pod logs to identify issues.
        
        Args:
            request: Task request containing logs and pod_name
            
        Returns:
            Task result containing analysis findings
        """
        try:
            # Extract parameters
            params = LogAnalysisRequest(**request.parameters)
            logs = params.logs
            pod_name = params.pod_name
            
            # Analyze logs
            analysis = await self.agent.analyze_logs(logs, pod_name)
            
            # Return the analysis
            return TaskResult(
                artifacts={
                    "findings": analysis,
                }
            )
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            return TaskResult(
                error=f"Failed to analyze logs: {str(e)}"
            )


async def main():
    """Run the Log Analysis Agent."""
    import uvicorn
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard
    
    # Create the agent executor
    executor = LogAnalysisAgentExecutor()
    
    # Create the agent card
    card = AgentCard(
        name="Log Analysis Agent",
        description="Analyzes Kubernetes pod logs to identify issues that might be causing CrashLoopBackOff states",
        capabilities=AgentCapabilities(
            can_stream_partial_response=True,
            can_handle_binary_data=False,
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            {
                "id": "analyze_logs",
                "name": "analyze_logs",
                "description": "Analyze Kubernetes pod logs to identify issues",
                "parameters": {
                    "logs": "string",
                    "pod_name": "string"
                },
                "tags": ["kubernetes", "logs", "analysis"]
            }
        ],
        url="http://localhost:8002",
        version="1.0.0"
    )
    
    # Create the task store
    task_store = InMemoryTaskStore()
    
    # Create the A2A application
    a2a_app = A2AStarletteApplication(
        http_handler=DefaultRequestHandler(agent_executor=executor, task_store=task_store),
        agent_card=card,
    )
    
    # Build the Starlette application
    app = a2a_app.build()
    
    # Run the server
    logger.info("Log Analysis Agent started")
    config = uvicorn.Config(app, host="0.0.0.0", port=8002)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
