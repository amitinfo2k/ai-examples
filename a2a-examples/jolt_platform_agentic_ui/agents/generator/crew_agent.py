from crewai import Agent, Task, Crew, Process, LLM
from agents.generator.tools import MCPReadFileTool
from langsmith import traceable
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith tracing for CrewAI using OpenInference instrumentation
# This is the recommended approach to get CrewAI traces into LangSmith
TRACING_ENABLED = False

if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
    try:
        # Step 1: Set up OpenTelemetry TracerProvider with LangSmith processor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from langsmith.integrations.otel import OtelSpanProcessor
        
        # Create tracer provider with LangSmith processor
        tracer_provider = TracerProvider()
        langsmith_processor = OtelSpanProcessor()
        tracer_provider.add_span_processor(langsmith_processor)
        trace.set_tracer_provider(tracer_provider)
        logger.info("OpenTelemetry TracerProvider configured with LangSmith processor")
        
        # Step 2: Instrument CrewAI with OpenInference (for agent/task tracing)
        try:
            from openinference.instrumentation.crewai import CrewAIInstrumentor
            CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
            logger.info("CrewAI instrumented -> LangSmith")
            TRACING_ENABLED = True
        except ImportError:
            logger.warning("openinference-instrumentation-crewai not installed")
        
        # Step 3: Instrument LiteLLM for LLM calls & token tracking
        # (CrewAI uses LiteLLM internally to call Gemini)
        try:
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
            LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
            logger.info("LiteLLM instrumented -> LangSmith (for token tracking)")
        except ImportError:
            logger.warning("openinference-instrumentation-litellm not installed")
        
        if TRACING_ENABLED:
            logger.info("LangSmith tracing is ENABLED for CrewAI")
            logger.info(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
        
    except ImportError as e:
        logger.warning(f"Tracing dependencies missing: {e}")
    except Exception as e:
        logger.error(f"Error setting up tracing: {e}")
else:
    logger.info("LangSmith tracing is DISABLED (set LANGCHAIN_TRACING_V2=true to enable)")



class JoltSpecGenerator:
    def __init__(self):
        self.mcp_tool = MCPReadFileTool()
        
        # Configure Gemini LLM
        # Check if API key is set, if not use a placeholder for demo
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY", "demo-key")
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini/gemini-2.0-flash-exp")
        
        logger.info(f"Initializing JoltSpecGenerator with model: {self.gemini_model}")
        
        self.llm = LLM(
            model=self.gemini_model,
            api_key=self.gemini_api_key
        )
        
    def create_agent(self):
        """Create the Jolt Spec Generator agent"""
        logger.info(f"Creating Jolt Spec Generator agent with model: {self.gemini_model}")
        return Agent(
            role='Jolt Specification Expert',
            goal='Generate accurate Jolt transformation specifications from input and output JSON examples',
            backstory="""You are an expert in JSON transformations and Jolt specifications.
            You can analyze input and output JSON structures and create precise Jolt specs
            that transform one into the other. You understand all Jolt operations: shift,
            default, remove, sort, cardinality, and modify.""",
            tools=[self.mcp_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_task(self, agent: Agent, input_path: str, output_path: str, auth_token: str):
        """Create the generation task"""
        logger.info("Creating Jolt Spec Generator task")
        return Task(
            description=f"""
            1. Read the input JSON from '{input_path}' using the read_file_from_drive tool with auth_token '{auth_token}'
            2. Read the output JSON from '{output_path}' using the read_file_from_drive tool with auth_token '{auth_token}'
            3. Analyze the structural differences between input and output
            4. Generate a Jolt specification that transforms the input into the output.
               IMPORTANT: Jolt 'shift' specs use the format "Input Path": "Output Path".
               - Left side (Key) is the path in the INPUT JSON.
               - Right side (Value) is the path in the OUTPUT JSON.
               - Do NOT invert this.
               - Use nesting for input structure traversal.
            5. Return ONLY the Jolt spec as a raw JSON array. Do not include 'Thought:', 'Action:', or any markdown formatting.
            
            The output must start with '[' and end with ']'.
            """,
            expected_output="A valid Jolt specification JSON array and nothing else",
            agent=agent
        )
    
    def create_refinement_task(self, agent: Agent, current_spec: dict, error_report: list):
        """Create the refinement task based on validation errors"""
        logger.info("Creating Jolt Spec Generator refinement task")
        # Format errors in a clear, actionable way
        error_summary = []
        for i, err in enumerate(error_report, 1):
            error_summary.append(f"Error {i}:")
            error_summary.append(f"  Path: {err.get('path', 'unknown')}")
            error_summary.append(f"  Expected: {err.get('expected', 'N/A')}")
            error_summary.append(f"  Actual: {err.get('actual', 'N/A')}")
            error_summary.append(f"  Issue: {err.get('error_description', 'Mismatch')}")
            error_summary.append("")
        
        return Task(
            description=f"""
            The Jolt specification you generated produced INCORRECT output.
            
            **Current (Broken) Spec:**
            {json.dumps(current_spec, indent=2)}
            
            **Validation Errors:**
            {chr(10).join(error_summary)}
            
            **Your Task:**
            You MUST fix the Jolt spec to correct these errors:
            
            1. **Analyze each error**: Understand which input field is mapped incorrectly.
            2. **Identify the root cause**: Is it the wrong source path, wrong target path, or wrong nesting?
            3. **Fix the spec**: Modify ONLY the problematic mappings in the spec.
            4. **Verify your changes**: Double-check that your new spec would produce the expected values.
            
            **CRITICAL RULES:**
            - If the error says "Value mismatch", check if you're reading from the correct INPUT field.
            - The LEFT side of a Jolt mapping is the INPUT path.
            - The RIGHT side is the OUTPUT path.
            - Do NOT repeat the same mistake.
            - Return ONLY the corrected JSON array, nothing else.
            
            **Output Format:**
            Start with '[' and end with ']'. NO explanations, NO markdown.
            """,
            expected_output="A corrected Jolt specification JSON array and nothing else",
            agent=agent
        )

    @traceable(name="crewai_refine_jolt_spec", run_type="chain")
    def refine_spec(self, current_spec: dict, error_report: list) -> dict:
        """Refine a Jolt spec based on error report"""
        logger.info(f"Starting refinement task with {len(error_report)} errors")
        agent = self.create_agent()
        logger.info("Creating Jolt Spec Generator refinement task")
        task = self.create_refinement_task(agent, current_spec, error_report)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True  # Enable verbose logging
        )
        
        logger.info("Kicking off CrewAI for refinement...")
        result = crew.kickoff()
        logger.info("CrewAI refinement completed")
        return self._parse_result(result)

    def create_prompt_refinement_task(self, agent: Agent, current_spec: dict, user_feedback: str, 
                                       input_json: dict = None, expected_output: dict = None,
                                       validation_errors: list = None):
        """Create a refinement task based on user's natural language feedback"""
        logger.info("Creating Jolt Spec Generator prompt-based refinement task")
        
        # Build context information
        context_parts = []
        
        if input_json:
            context_parts.append(f"**Input JSON:**\n```json\n{json.dumps(input_json, indent=2)}\n```")
        
        if expected_output:
            context_parts.append(f"**Expected Output JSON:**\n```json\n{json.dumps(expected_output, indent=2)}\n```")
        
        if validation_errors:
            error_summary = []
            for i, err in enumerate(validation_errors, 1):
                if isinstance(err, dict):
                    error_summary.append(f"Error {i}:")
                    error_summary.append(f"  Path: {err.get('path', 'unknown')}")
                    error_summary.append(f"  Expected: {err.get('expected', 'N/A')}")
                    error_summary.append(f"  Actual: {err.get('actual', 'N/A')}")
                    error_summary.append(f"  Issue: {err.get('error_description', 'Mismatch')}")
                    error_summary.append("")
            if error_summary:
                context_parts.append(f"**Previous Validation Errors:**\n{chr(10).join(error_summary)}")
        
        context_section = chr(10).join(context_parts) if context_parts else "No additional context provided."
        
        return Task(
            description=f"""
            A user is providing feedback to help you fix a Jolt specification.
            
            **Current Jolt Spec (needs fixing):**
            ```json
            {json.dumps(current_spec, indent=2)}
            ```
            
            **Context Information:**
            {context_section}
            
            **User's Feedback/Instructions:**
            "{user_feedback}"
            
            **Your Task:**
            Based on the user's feedback, modify the Jolt specification to address their concerns.
            
            1. **Carefully read the user's feedback** - They may be pointing out specific issues or giving hints about what fields should map where.
            2. **Consider the context** - Look at the input/output JSON and any validation errors to understand what's going wrong.
            3. **Apply the fix** - Modify the Jolt spec according to the user's guidance.
            4. **Validate your changes** - Make sure the modified spec would transform the input to match the expected output.
            
            **CRITICAL RULES:**
            - The LEFT side of a Jolt mapping is the INPUT path (where to read from).
            - The RIGHT side is the OUTPUT path (where to write to).
            - Return ONLY the corrected JSON array, nothing else.
            - Start with '[' and end with ']'. NO explanations, NO markdown.
            """,
            expected_output="A corrected Jolt specification JSON array and nothing else",
            agent=agent
        )

    @traceable(name="crewai_refine_jolt_spec_with_prompt", run_type="chain")
    def refine_spec_with_prompt(self, current_spec: dict, user_feedback: str,
                                 input_json: dict = None, expected_output: dict = None,
                                 validation_errors: list = None) -> dict:
        """Refine a Jolt spec based on user's natural language feedback"""
        logger.info(f"Starting prompt-based refinement with user feedback: {user_feedback[:100]}...")
        agent = self.create_agent()
        task = self.create_prompt_refinement_task(
            agent, current_spec, user_feedback, 
            input_json, expected_output, validation_errors
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("Kicking off CrewAI for prompt-based refinement...")
        result = crew.kickoff()
        logger.info("CrewAI prompt-based refinement completed")
        return self._parse_result(result)

    def _parse_result(self, result):
        """Helper to parse JSON result"""
        logger.info("Parsing result")
        try:
            result_str = str(result)
            
            # clean up potential markdown code blocks
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0]
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0]
            
            # If there's extra text after the JSON (like "Thought: ..."), try to extract just the JSON list
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result_str, re.DOTALL)
            
            if json_match:
                result_str = json_match.group(0)
            
            spec = json.loads(result_str.strip())
            return spec
        except Exception as e:
            return {"error": f"Failed to parse Jolt spec: {str(e)}", "raw_output": str(result)}

    @traceable(name="crewai_generate_jolt_spec", run_type="chain")
    def generate(self, input_path: str, output_path: str, auth_token: str) -> dict:
        """Generate a Jolt spec from input and output files"""
        logger.info(f"Starting generation task for input={input_path}, output={output_path}")
        agent = self.create_agent()
        task = self.create_task(agent, input_path, output_path, auth_token)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True  # Enable verbose logging
        )
        
        logger.info("Kicking off CrewAI for generation...")
        result = crew.kickoff()
        logger.info("CrewAI generation completed")
        return self._parse_result(result)

