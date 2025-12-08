from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, END
from deepdiff import DeepDiff
import json
import uuid
import os

from agents.validator.jolt_utils import apply_jolt_shift_async
from agents.validator.a2a_protocol import (
    A2AProtocol, ErrorReport, VerificationResult
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith tracing if enabled
if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
    try:
        from langsmith import Client
        from langchain_core.tracers import LangChainTracer
        
        # Create LangSmith tracer for LangGraph
        langsmith_tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "jolt-platform")
        )
        logger.info("LangSmith tracing is ENABLED for LangGraph Validator")
        logger.info(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
        TRACING_ENABLED = True
    except ImportError as e:
        logger.warning(f"LangSmith tracing requested but dependencies not installed: {e}")
        langsmith_tracer = None
        TRACING_ENABLED = False
else:
    logger.info("LangSmith tracing is DISABLED")
    langsmith_tracer = None
    TRACING_ENABLED = False


class ValidatorState(TypedDict):
    """State for the validator workflow"""
    input_json: Dict[str, Any]
    expected_output: Dict[str, Any]
    jolt_spec: Dict[str, Any]
    actual_output: Dict[str, Any]
    validation_result: VerificationResult
    error_reports: list
    logs: List[str]
    conversation_id: str
    a2a_protocol: A2AProtocol

class JoltValidator:
    def __init__(self):
        self.protocol = A2AProtocol()
        # Get generator service URL for A2A communication
        import os
        self.generator_url = os.getenv("GENERATOR_URL", "http://localhost:8081")
        logger.info(f"JoltValidator initialized with Generator URL: {self.generator_url}")
    
    async def validate_spec(
        self,
        input_json: Dict[str, Any],
        expected_output: Dict[str, Any],
        jolt_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main validation entry point
        
        Args:
            input_json: Input JSON data
            expected_output: Expected output JSON
            jolt_spec: Jolt specification to validate
            
        Returns:
            Dictionary with validation results
        """
        # Create initial state
        conversation_id = f"jolt_validate_{uuid.uuid4()}"
        initial_state = {
            "input_json": input_json,
            "expected_output": expected_output,
            "jolt_spec": jolt_spec,
            "actual_output": {},
            "validation_result": None,
            "error_reports": [],
            "logs": ["Starting validation..."],
            "conversation_id": conversation_id,
            "a2a_protocol": self.protocol
        }
        
        # Create the workflow
        workflow = StateGraph(ValidatorState)
        
        # Add nodes
        workflow.add_node("transform", self.transform_node)
        workflow.add_node("compare", self.compare_node)
        workflow.add_node("analyze_errors", self.analyze_errors_node)
        workflow.add_conditional_edges(
            "compare",
            self.decide_next_step,
            {
                "analyze_errors": "analyze_errors",
                "end": END
            }
        )
        workflow.add_edge("analyze_errors", END) # End after analysis
        workflow.add_edge("transform", "compare")
        
        # Set entry point
        workflow.set_entry_point("transform")
        
        # Compile the workflow
        chain = workflow.compile()
        
        # Run the workflow
        final_state = await chain.ainvoke(initial_state)
        
        return {
            "is_valid": final_state["validation_result"].is_valid if final_state["validation_result"] else False,
            "errors": final_state["error_reports"],
            "actual_output": final_state["actual_output"],
            "a2a_messages": self.protocol.get_conversation_history(),
            "logs": final_state["logs"]
        }
    
    async def validate_with_retries(
        self,
        input_json: Dict[str, Any],
        expected_output: Dict[str, Any],
        jolt_spec: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Validate with A2A refinement loop - implements collaborative debugging using ADK Task Lifecycle
        """
        import httpx
        import asyncio
        
        current_spec = jolt_spec
        all_logs = []
        
        for attempt in range(max_retries + 1):
            logger.info(f"Validation attempt {attempt + 1}/{max_retries + 1}")
            all_logs.append(f"--- Validation Attempt {attempt + 1}/{max_retries + 1} ---")
            
            # Validate current spec
            validation_result = await self.validate_spec(
                input_json=input_json,
                expected_output=expected_output,
                jolt_spec=current_spec
            )
            
            # Merge logs
            all_logs.extend(validation_result.get("logs", []))
            
            if validation_result["is_valid"]:
                logger.info(f"Validation successful on attempt {attempt + 1}")
                return {
                    "is_valid": True,
                    "jolt_spec": current_spec,
                    "actual_output": validation_result["actual_output"],
                    "errors": [],
                    "logs": all_logs,
                    "attempts": attempt + 1,
                    "a2a_messages": validation_result.get("a2a_messages", [])
                }
            
            # If not valid and we have retries left, initiate A2A refinement
            if attempt < max_retries:
                logger.info(f"Validation failed. Initiating A2A refinement with Generator...")
                all_logs.append(f"Validation failed with {len(validation_result['errors'])} errors. Requesting refinement from Generator...")
                
                try:
                    # Convert ErrorReport objects to dictionaries for JSON serialization
                    errors_to_send = []
                    for err in validation_result["errors"]:
                        if hasattr(err, 'model_dump'):
                            errors_to_send.append(err.model_dump())
                        elif hasattr(err, 'dict'):
                            errors_to_send.append(err.dict())
                        else:
                            errors_to_send.append(err)
                    
                    # ADK Task Lifecycle: 1. Submit Task
                    async with httpx.AsyncClient(timeout=300.0) as client:
                        # Discovery Phase: Fetch Agent Card
                        try:
                            discovery_url = f"{self.generator_url}/.well-known/agent.json"
                            logger.info(f"Discovery: Fetching Agent Card from {discovery_url}")
                            agent_card_resp = await client.get(discovery_url)
                            if agent_card_resp.status_code == 200:
                                agent_card = agent_card_resp.json()
                                # Log Discovery Interaction
                                self.protocol.log_interaction(
                                    source="validator",
                                    target="generator",
                                    action="discovery",
                                    details=agent_card
                                )
                                all_logs.append(f"Discovered Agent: {agent_card.get('name')} ({discovery_url})")
                        except Exception as e:
                            logger.warning(f"Discovery failed: {e}")
                        
                        logger.info(f"Submitting Refinement Task to Generator at {self.generator_url}/tasks")
                        response = await client.post(
                            f"{self.generator_url}/tasks",
                            json={
                                "task_type": "refine",
                                "input_data": {
                                    "current_spec": current_spec,
                                    "error_report": errors_to_send
                                }
                            }
                        )
                        response.raise_for_status()
                        task_info = response.json()
                        task_id = task_info["task_id"]
                        logger.info(f"Task submitted. Task ID: {task_id}")
                        all_logs.append(f"Refinement task submitted (ID: {task_id}). Waiting for completion...")
                        
                        # ADK Task Lifecycle: 2. Poll for Completion
                        # (In production, use webhooks or async events if supported)
                        max_poll_attempts = 60 # 5 minutes max (5s interval)
                        poll_interval = 5
                        
                        for _ in range(max_poll_attempts):
                            await asyncio.sleep(poll_interval)
                            status_resp = await client.get(f"{self.generator_url}/tasks/{task_id}")
                            status_resp.raise_for_status()
                            task_status = status_resp.json()
                            
                            if task_status["status"] == "completed":
                                logger.info("Task completed successfully")
                                refine_result = task_status["output"]
                                current_spec = refine_result.get("jolt_spec")
                                all_logs.append("Refinement task completed. Received new spec.")
                                
                                # Log the interaction (Generator -> Validator)
                                self.protocol.log_interaction(
                                    source="generator",
                                    target="validator",
                                    action="patch_proposal",
                                    details={"jolt_spec": current_spec}
                                )
                                break
                            elif task_status["status"] == "failed":
                                error_msg = task_status.get("error", "Unknown error")
                                raise Exception(f"Generator task failed: {error_msg}")
                            
                            logger.info(f"Task status: {task_status['status']}...")
                        else:
                            raise Exception("Task polling timed out")
                        
                except httpx.HTTPError as e:
                    logger.error(f"A2A communication failed: {str(e)}")
                    all_logs.append(f"A2A refinement failed: {str(e)}")
                    break
                except Exception as e:
                    logger.error(f"Error during A2A refinement: {str(e)}")
                    all_logs.append(f"A2A refinement error: {str(e)}")
                    break
            else:
                logger.info(f"Max retries ({max_retries}) reached. Validation failed.")
                all_logs.append(f"Max retries reached. Validation unsuccessful.")
        
        # Return final result (unsuccessful)
        return {
            "is_valid": False,
            "jolt_spec": current_spec,
            "actual_output": validation_result.get("actual_output", {}),
            "errors": validation_result.get("errors", []),
            "logs": all_logs,
            "attempts": max_retries + 1,
            "a2a_messages": validation_result.get("a2a_messages", [])
        }
    
    async def transform_node(self, state: ValidatorState) -> ValidatorState:
        """Apply Jolt transformation"""
        logger.info("Starting Jolt transformation node")
        try:
            # Convert jolt_spec to list format if it's not already
            spec = state["jolt_spec"]
            if isinstance(spec, dict) and "spec" not in spec:
                # Wrap in operation format
                spec = [{"operation": "shift", "spec": spec}]
            elif isinstance(spec, dict):
                spec = [spec]
            
            logger.info(f"Applying Jolt spec: {json.dumps(spec)[:100]}...")
            # Use the async version of apply_jolt_shift
            actual_output = await apply_jolt_shift_async(state["input_json"], spec)
            logger.info("Jolt transformation completed successfully")
            state["actual_output"] = actual_output
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            state["actual_output"] = {"error": f"Transformation failed: {str(e)}"}
            state["logs"].append(f"Transformation Error: {str(e)}")
        
        return state
    
    def compare_node(self, state: ValidatorState) -> ValidatorState:
        """Compare actual vs expected output"""
        logger.info("Starting comparison node")
        logs = state.get("logs", [])
        logs.append("--- Validation Comparison ---")
        
        try:
            logs.append(f"Expected Output: {json.dumps(state['expected_output'], indent=2)}")
            logs.append(f"Actual Output:   {json.dumps(state['actual_output'], indent=2)}")
            
            diff = DeepDiff(state["expected_output"], state["actual_output"], ignore_order=True)
            
            if not diff:
                logger.info("Validation successful: MATCH")
                logs.append("Result: MATCH ✅")
                state["validation_result"] = VerificationResult(
                    is_valid=True,
                    errors=[],
                    success_message="Jolt spec is valid. Transformation successful!"
                )
                
                # Send success message via A2A
                state["a2a_protocol"].log_interaction(
                    source="validator",
                    target="generator",
                    action="verification_result",
                    details=state["validation_result"].model_dump()
                )
            else:
                logger.info(f"Validation failed: MISMATCH. Diff: {diff}")
                logs.append(f"Result: MISMATCH ❌\nDiff: {diff}")
                errors = self.diff_to_errors(diff)
                state["error_reports"] = errors
                state["validation_result"] = VerificationResult(
                    is_valid=False,
                    errors=errors
                )
        except Exception as e:
            logger.error(f"Comparison error: {str(e)}")
            logs.append(f"Result: ERROR ❌\nException: {str(e)}")
            state["validation_result"] = VerificationResult(
                is_valid=False,
                errors=[ErrorReport(
                    path="root",
                    expected="Successful comparison",
                    actual=f"Exception: {str(e)}",
                    error_description="Comparison failed"
                )]
            )
            
        state["logs"] = logs
        return state
    
    async def analyze_errors_node(self, state: ValidatorState) -> ValidatorState:
        """Analyze errors using LLM and send an A2A error report."""
        logger.info(f"Analyzing {len(state['error_reports'])} errors with LLM")
        
        try:
            # Initialize Gemini LLM
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-pro"),
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Construct prompt for analysis
            prompt = f"""
            You are an expert Jolt transformation validator.
            
            Analyze the following validation failure:
            
            Input JSON:
            {json.dumps(state['input_json'], indent=2)}
            
            Expected Output:
            {json.dumps(state['expected_output'], indent=2)}
            
            Actual Output:
            {json.dumps(state['actual_output'], indent=2)}
            
            Current Jolt Spec:
            {json.dumps(state['jolt_spec'], indent=2)}
            
            Errors Found:
            {json.dumps([e.model_dump() for e in state['error_reports']], indent=2)}
            
            Provide a concise but technical analysis of why the transformation failed and what specific changes are needed in the Jolt spec to fix it.
            Focus on the Jolt operations (shift, default, etc.) and path matching.
            """
            
            # Get analysis from LLM
            response = await llm.ainvoke(prompt)
            analysis = response.content
            
            # Add analysis to the first error report or create a summary report
            if state["error_reports"]:
                state["error_reports"][0].error_description += f"\n\nAI Analysis:\n{analysis}"
            
            logger.info("LLM analysis completed and added to error report")
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            # Continue without LLM analysis
        
        # Send error report to Generator
        state["a2a_protocol"].log_interaction(
            source="validator",
            target="generator",
            action="error_report",
            details=[e.model_dump() for e in state["error_reports"]]
        )
        return state

    def decide_next_step(self, state: ValidatorState) -> str:
        """Determine if we should end or proceed to error analysis."""
        if state["validation_result"] and state["validation_result"].is_valid:
            return "end"
        else:
            return "analyze_errors"
    
    def diff_to_errors(self, diff: DeepDiff) -> list:
        """Convert DeepDiff result to ErrorReport list"""
        errors = []
        
        # Handle dictionary items added (present in actual but not expected)
        if 'dictionary_item_added' in diff:
            for path in diff['dictionary_item_added']:
                errors.append(ErrorReport(
                    path=str(path),
                    expected="Not present",
                    actual="Present",
                    error_description="Unexpected item in output"
                ))
                
        # Handle dictionary items removed (present in expected but not actual)
        if 'dictionary_item_removed' in diff:
            for path in diff['dictionary_item_removed']:
                errors.append(ErrorReport(
                    path=str(path),
                    expected="Present",
                    actual="Missing",
                    error_description="Missing item in output"
                ))
                
        # Handle value changes
        if 'values_changed' in diff:
            for path, change in diff['values_changed'].items():
                errors.append(ErrorReport(
                    path=str(path),
                    expected=change.get("old_value"), # DeepDiff 'old' is usually the first arg (expected)
                    actual=change.get("new_value"),   # DeepDiff 'new' is usually the second arg (actual)
                    error_description=f"Value mismatch at {path}"
                ))
        
        return errors
