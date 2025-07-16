"""
User Agent implementation.

This agent serves as the interface for users to interact with the system
and initiate the pod troubleshooting process.
"""

import asyncio
import logging
from typing import Dict, Any, AsyncGenerator

from a2a.client import A2AClient
from a2a.types import TextPart
from pydantic import BaseModel, Field

from common.base_agent import BaseAgent
from common.agent_executor import GenericAgentExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define orchestrator agent URL
ORCHESTRATOR_AGENT_URL = "http://localhost:8003"


class TroubleshootPodRequest(BaseModel):
    """Request model for pod troubleshooting."""
    namespace: str
    pod_name: str


class UserAgent(BaseAgent):
    """Agent that serves as the user interface."""
    
    agent_name: str = Field(default="User Agent")
    description: str = Field(
        default="Interface for users to interact with the system and initiate pod troubleshooting"
    )
    content_types: list[str] = Field(default=["text/plain"])
    
    async def stream(self, query: str, context_id: str, task_id: str) -> AsyncGenerator[TextPart, None]:
        """Stream responses from the agent."""
        yield TextPart(text=f"Received query: {query}\nProcessing user request...")
        
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
                    yield TextPart(text=f"Initiating troubleshooting for pod {pod_name} in namespace {namespace}...")
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
                yield TextPart(text="I'm the User Agent, your interface to the Kubernetes troubleshooting system.")
                yield TextPart(text="You can ask me to troubleshoot a pod in a namespace using the format: Troubleshoot pod <pod_name> in namespace <namespace>")
        except Exception as e:
            yield TextPart(text=f"Error processing query: {str(e)}")
            import traceback
            yield TextPart(text=f"Traceback: {traceback.format_exc()}")
        
        yield TextPart(text="Query processing complete.")
        
    
    async def troubleshoot_pod(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        """
        Initiate troubleshooting for a pod in CrashLoopBackOff state.
        
        Args:
            namespace: The namespace of the pod
            pod_name: The name of the pod
            
        Returns:
            Troubleshooting report
        """
        try:
            logger.info(f"Initiating troubleshooting for pod {pod_name} in namespace {namespace}")
            
            # Create a client to communicate with the orchestrator agent
            orchestrator_client = A2AClient(ORCHESTRATOR_AGENT_URL)
            
            # Send request to orchestrator agent
            response = await orchestrator_client.invoke_skill(
                "troubleshoot_pod",
                {"namespace": namespace, "pod_name": pod_name},
            )
            
            if response.get("error"):
                return {"error": f"Troubleshooting failed: {response['error']}"}
            
            # Return the troubleshooting report
            return {"report": response.get("report", "No report available")}
            
        except Exception as e:
            logger.error(f"Error troubleshooting pod: {e}")
            return {"error": f"Failed to troubleshoot pod: {str(e)}"}


async def main():
    """Run the User Agent."""
    import uvicorn
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard
    
    # Create the agent
    agent = UserAgent(
        agent_name="User Agent",
        description="Interface for users to interact with the system and initiate pod troubleshooting",
        content_types=["text/plain"]
    )
    
    # Create the agent executor
    executor = GenericAgentExecutor(agent)
    
    # Create the agent card
    card = AgentCard(
        name="User Agent",
        description="Interface for users to interact with the system and initiate pod troubleshooting",
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
                "description": "Initiate troubleshooting for a pod in CrashLoopBackOff state",
                "parameters": {
                    "namespace": "string",
                    "pod_name": "string"
                },
                "tags": ["kubernetes", "troubleshooting", "pod"]
            }
        ],
        url="http://localhost:8004",
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
    logger.info("User Agent started")
    config = uvicorn.Config(app, host="0.0.0.0", port=8004)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
