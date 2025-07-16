"""
Orchestrator Agent implementation.

This agent acts as the central coordinator, delegating tasks to specialized agents
and aggregating their findings to diagnose Kubernetes pod CrashLoopBackOff issues.
"""

import asyncio
import logging
import os
from typing import Dict, Any, AsyncGenerator

import google.generativeai as genai
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import TextPart
from a2a.types import Task as TaskRequest
from common.types import TaskResult
from a2a.utils import new_agent_text_message
from a2a.client import A2AClient
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

# Define agent URLs
K8S_AGENT_URL = "http://localhost:8001"
LOG_AGENT_URL = "http://localhost:8002"


class TroubleshootRequest(BaseModel):
    """Request model for pod troubleshooting."""
    namespace: str
    pod_name: str


class OrchestratorAgent(BaseAgent):
    """Implementation of the Orchestrator Agent."""
    
    agent_name: str = Field(default="Orchestrator Agent")
    description: str = Field(
        default="Central coordinator that delegates tasks to specialized agents and aggregates findings"
    )
    content_types: list[str] = Field(default=["text/plain"])
    
    async def stream(self, query: str, context_id: str, task_id: str) -> AsyncGenerator[TextPart, None]:
        """Stream responses from the agent."""
        yield TextPart(text=f"Received query: {query}\nProcessing orchestration request...")
        
        try:
            # Simple query parsing - this could be improved with NLP or a more sophisticated parser
            query_lower = query.lower()
            
            if "troubleshoot" in query_lower and "pod" in query_lower:
                # Extract namespace and pod name from query
                parts = query_lower.split()
                namespace = None
                pod_name = None
                
                for i, part in enumerate(parts):
                    if part == "namespace" and i + 1 < len(parts):
                        namespace = parts[i + 1]
                    if part == "pod" and i + 1 < len(parts):
                        pod_name = parts[i + 1]
                
                if namespace and pod_name:
                    yield TextPart(text=f"Troubleshooting pod {pod_name} in namespace {namespace}...")
                    try:
                        result = await self.troubleshoot_pod(namespace, pod_name)
                        if "error" in result:
                            yield TextPart(text=f"Error: {result['error']}")
                        else:
                            yield TextPart(text=f"Troubleshooting report: {result['report']}")
                    except Exception as e:
                        yield TextPart(text=f"Error troubleshooting pod: {str(e)}")
                else:
                    yield TextPart(text="Could not determine namespace and pod name from query. Please specify both.")
                    yield TextPart(text="Format your request as: Troubleshoot pod <pod_name> in namespace <namespace>")
            else:
                yield TextPart(text="I'm not sure what you're asking for. You can ask me to troubleshoot a pod in a namespace.")
                yield TextPart(text="Format your request as: Troubleshoot pod <pod_name> in namespace <namespace>")
        except Exception as e:
            yield TextPart(text=f"Error processing query: {str(e)}")
            import traceback
            yield TextPart(text=f"Traceback: {traceback.format_exc()}")
        
        yield TextPart(text="Query processing complete.")
        

    async def troubleshoot_pod(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        """Troubleshoot a Kubernetes pod in CrashLoopBackOff state."""
        logger.info(f"Troubleshooting pod {pod_name} in namespace {namespace}")
        
        try:
            # Step 1: Get pod information from Kubernetes Info Agent
            k8s_client = A2AClient(K8S_AGENT_URL)
            pod_info_response = await k8s_client.invoke_skill(
                "retrieve_k8s_info",
                {"namespace": namespace, "pod_name": pod_name}
            )
            
            if pod_info_response.get("error"):
                return {"error": f"Failed to get pod info: {pod_info_response['error']}"}
            
            pod_yaml = pod_info_response.get("pod_yaml", "")
            pod_events = pod_info_response.get("pod_events", {})
            
            # Step 2: Get pod logs from Kubernetes Info Agent
            pod_logs_response = await k8s_client.invoke_skill(
                "get_pod_logs",
                {"namespace": namespace, "pod_name": pod_name}
            )
            
            if pod_logs_response.get("error"):
                logger.warning(f"Failed to get pod logs: {pod_logs_response['error']}")
                pod_logs = "No logs available"
            else:
                pod_logs = pod_logs_response.get("logs", "No logs available")
            
            # Step 3: Analyze logs with Log Analysis Agent
            log_client = A2AClient(LOG_AGENT_URL)
            log_analysis_response = await log_client.invoke_skill(
                "analyze_logs",
                {"logs": pod_logs, "pod_name": pod_name}
            )
            
            if log_analysis_response.get("error"):
                logger.warning(f"Failed to analyze logs: {log_analysis_response['error']}")
                log_findings = "Log analysis failed"
            else:
                log_findings = log_analysis_response.get("analysis", "No analysis available")
            
            # Step 4: Synthesize findings and generate recommendations
            report = await self._synthesize_findings(
                namespace, pod_name, pod_yaml, pod_events, log_findings
            )
            
            return {"report": report}
            
        except Exception as e:
            logger.error(f"Error troubleshooting pod: {e}")
            return {"error": f"Failed to troubleshoot pod: {str(e)}"}

class OrchestratorAgentExecutor(GenericAgentExecutor):
    """Executor for the Orchestrator Agent."""

    def __init__(self):
        """Initialize the Orchestrator Agent Executor."""
        agent = OrchestratorAgent(
            agent_name="Orchestrator Agent",
            description="Central coordinator that delegates tasks to specialized agents and aggregates findings",
            content_types=["text/plain"]
        )
        super().__init__(agent)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the agent execution."""
        await event_queue.enqueue_event(new_agent_text_message("Orchestrator Agent task cancelled"))
    
    async def troubleshoot_pod(self, request: TaskRequest) -> TaskResult:
        """
        Troubleshoot a Kubernetes pod in CrashLoopBackOff state.
        
        Args:
            request: Task request containing namespace and pod_name
            
        Returns:
            Task result containing troubleshooting report and recommendations
        """
        try:
            # Extract parameters
            params = TroubleshootRequest(**request.parameters)
            namespace = params.namespace
            pod_name = params.pod_name
            
            # Troubleshoot the pod
            result = await self.agent.troubleshoot_pod(namespace, pod_name)
            
            if "error" in result:
                return TaskResult(error=result["error"])
            
            # Return the troubleshooting report
            return TaskResult(
                artifacts={
                    "report": result["report"],
                }
            )
        except Exception as e:
            logger.error(f"Error troubleshooting pod: {e}")
            return TaskResult(
                error=f"Failed to troubleshoot pod: {str(e)}"
            )

    async def _synthesize_findings(
        self,
        namespace: str,
        pod_name: str,
        pod_yaml: str,
        pod_events: Dict[str, Any],
        log_findings: str,
    ) -> str:
        """
        Synthesize findings from different sources to create a comprehensive report.
        
        Args:
            namespace: The namespace of the pod
            pod_name: The name of the pod
            pod_yaml: The pod YAML description
            pod_events: The pod events
            log_findings: Findings from log analysis
            
        Returns:
            A comprehensive troubleshooting report
        """
        # Format events for better readability
        formatted_events = "No events found"
        if pod_events and isinstance(pod_events, dict) and "items" in pod_events:
            events_list = pod_events.get("items", [])
            if events_list:
                formatted_events = "\n".join(
                    [
                        f"- Type: {event.get('type', 'Unknown')}, "
                        f"Reason: {event.get('reason', 'Unknown')}, "
                        f"Message: {event.get('message', 'No message')}, "
                        f"Count: {event.get('count', 0)}, "
                        f"First Timestamp: {event.get('firstTimestamp', 'Unknown')}, "
                        f"Last Timestamp: {event.get('lastTimestamp', 'Unknown')}"
                        for event in events_list
                    ]
                )
        
        # Prepare prompt for Gemini
        prompt = f"""
        Analyze the following Kubernetes pod YAML, events, and log analysis findings to determine 
        the root cause of the 'CrashLoopBackOff' state for pod '{pod_name}' in namespace '{namespace}'.
        Provide a concise explanation of the problem and actionable remediation steps.
        
        Pod YAML:
        {pod_yaml}
        
        Pod Events:
        {formatted_events}
        
        Log Analysis Findings:
        {log_findings}
        
        Your analysis should include:
        1. A clear identification of the root cause
        2. Evidence supporting your conclusion
        3. Step-by-step remediation instructions
        4. Any preventative measures for the future
        
        Format your response as a structured report with sections for:
        - Summary
        - Root Cause Analysis
        - Evidence
        - Remediation Steps
        - Prevention Recommendations
        """
        
        # Call Gemini for synthesis
        if not GEMINI_API_KEY:
            # Mock response for testing without API key
            return """
            # Pod Troubleshooting Report
            
            ## Summary
            The pod is experiencing a CrashLoopBackOff due to resource constraints.
            
            ## Root Cause Analysis
            This is a mock analysis since the Gemini API key is not configured.
            
            ## Evidence
            Mock evidence from pod events and logs.
            
            ## Remediation Steps
            1. Increase memory limits in the pod specification
            2. Restart the deployment
            
            ## Prevention Recommendations
            Implement proper resource monitoring and scaling.
            """
        else:
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                return f"Error generating report: {str(e)}"


async def main():
    """Run the Orchestrator Agent."""
    import uvicorn
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard
    
    # Create the agent executor
    executor = OrchestratorAgentExecutor()
    
    # Create the agent card
    card = AgentCard(
        name="Orchestrator Agent",
        description="Central coordinator that delegates tasks to specialized agents and aggregates findings",
        capabilities=AgentCapabilities(
            can_stream_partial_response=True,
            can_handle_binary_data=False,
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            {
                "id": "troubleshoot_pod",
                "name": "troubleshoot_pod",
                "description": "Troubleshoot a Kubernetes pod in CrashLoopBackOff state",
                "parameters": {
                    "namespace": "string",
                    "pod_name": "string"
                },
                "tags": ["kubernetes", "troubleshooting", "pod"]
            }
        ],
        url="http://localhost:8003",
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
    logger.info("Orchestrator Agent started")
    config = uvicorn.Config(app, host="0.0.0.0", port=8003)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
