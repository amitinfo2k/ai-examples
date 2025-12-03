from crewai import Agent, Task, Crew, Process, LLM
from agents.generator.tools import MCPReadFileTool
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

