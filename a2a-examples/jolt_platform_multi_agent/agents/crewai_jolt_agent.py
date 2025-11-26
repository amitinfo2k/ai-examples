"""
CrewAI Agent for JOLT Specification Generation
This agent is responsible for creating JOLT transformation specifications
based on input and output JSON examples.
Uses Google Gemini for LLM capabilities.
"""

from crewai import Agent, Task, Crew, LLM
from typing import Dict, Any
import json
import os


class JoltSpecificationCreator:
    """CrewAI-based agent for creating JOLT specifications using Google Gemini."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the JOLT Specification Creator agent with Google Gemini.
        
        Args:
            model_name: Gemini model to use (defaults to gemini-1.5-pro)
        """
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        
        # Configure Gemini LLM for CrewAI
        self.llm = LLM(
            model=f"gemini/{self.model_name}",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent for JOLT specification generation."""
        return Agent(
            role='JOLT Specification Engineer',
            goal='Create accurate and efficient JOLT transformation specifications',
            backstory="""You are an expert in JOLT (JSON to JSON transformation) specifications.
            You understand JSON data structures deeply and can create precise JOLT specs that
            transform input JSON to the desired output JSON format. You are familiar with all
            JOLT operations: shift, default, remove, sort, cardinality, and modify.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm  # Use Gemini LLM
        )
    
    def create_jolt_spec(self, input_json: Dict[str, Any], output_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a JOLT specification based on input and output JSON examples.
        
        Args:
            input_json: The source JSON structure
            output_json: The desired target JSON structure
            
        Returns:
            A JOLT specification as a dictionary
        """
        task = Task(
            description=f"""
            Create a JOLT specification that transforms the following input JSON 
            to the output JSON format.
            
            INPUT JSON:
            {json.dumps(input_json, indent=2)}
            
            EXPECTED OUTPUT JSON:
            {json.dumps(output_json, indent=2)}
            
            Requirements:
            1. Analyze both JSON structures carefully
            2. Identify all field mappings, transformations, and restructuring needed
            3. Create a valid JOLT specification using appropriate operations
            4. Ensure the spec handles nested structures correctly
            5. Return ONLY the JOLT specification as valid JSON
            6. Use shift, default, and other JOLT operations as needed
            
            Return the JOLT specification in the following format:
            [
              {{
                "operation": "shift",
                "spec": {{
                  // your shift mappings here
                }}
              }},
              // additional operations if needed
            ]
            """,
            agent=self.agent,
            expected_output="A valid JOLT specification in JSON format"
        )
        
        # Create a crew with the agent and task
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False
        )
        
        # Execute the task
        result = crew.kickoff()
        
        # Parse the result
        try:
            # Try to extract JSON from the result
            result_str = str(result)
            
            # Find JSON array in the result
            start_idx = result_str.find('[')
            end_idx = result_str.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                jolt_spec = json.loads(result_str[start_idx:end_idx])
                return jolt_spec
            else:
                # Try to parse the entire result
                jolt_spec = json.loads(result_str)
                return jolt_spec
        except json.JSONDecodeError as e:
            print(f"Error parsing JOLT spec: {e}")
            print(f"Raw result: {result}")
            raise ValueError(f"Failed to parse JOLT specification from agent output: {e}")
    
    def create_jolt_spec_from_files(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Create a JOLT specification from input and output JSON files.
        
        Args:
            input_file: Path to the input JSON file
            output_file: Path to the expected output JSON file
            
        Returns:
            A JOLT specification as a dictionary
        """
        with open(input_file, 'r') as f:
            input_json = json.load(f)
        
        with open(output_file, 'r') as f:
            output_json = json.load(f)
        
        return self.create_jolt_spec(input_json, output_json)
    
    def save_jolt_spec(self, jolt_spec: Dict[str, Any], output_file: str):
        """
        Save the JOLT specification to a file.
        
        Args:
            jolt_spec: The JOLT specification to save
            output_file: Path to save the specification
        """
        with open(output_file, 'w') as f:
            json.dump(jolt_spec, f, indent=2)
        print(f"JOLT specification saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Sample input and output
    sample_input = {
        "user": {
            "firstName": "John",
            "lastName": "Doe",
            "email": "john.doe@example.com"
        },
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    sample_output = {
        "fullName": "John Doe",
        "contact": {
            "email": "john.doe@example.com"
        },
        "eventtime": "2024-01-01T12:00:00Z"
    }
    
    # Create the agent
    creator = JoltSpecificationCreator()
    
    # Generate JOLT spec
    jolt_spec = creator.create_jolt_spec(sample_input, sample_output)
    
    # Save the spec
    creator.save_jolt_spec(jolt_spec, "jolt_spec.json")
    
    print("\nGenerated JOLT Specification:")
    print(json.dumps(jolt_spec, indent=2))
