import json
import os
import logging
from typing import Dict, Any, List, Optional

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from jolt_ocsf_parser_ai.tools import JoltTransformTool

# Configure logging
logger = logging.getLogger(__name__)

# Configure logging
logger = logging.getLogger(__name__)

@CrewBase
class JoltOcsfParserAi():
    """JoltOcsfParserAi crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def ocsf_jolt_parser(self) -> Agent:
        """Create the OCSF to JOLT parser agent using configuration from agents.yaml."""
        agent = self._create_agent('ocsf_jolt_parser')
        # Ensure the agent has the JoltTransformTool
        agent.tools = [JoltTransformTool()]
        return agent
        
    @agent
    def jolt_spec_validator(self) -> Agent:
        """Create the JOLT spec validator agent using configuration from agents.yaml."""
        agent = self._create_agent('jolt_spec_validator')
        # Ensure the agent has the JoltTransformTool
        agent.tools = [JoltTransformTool()]
        return agent
        
    def _create_agent(self, agent_name: str) -> Agent:
        """Helper method to create an agent from configuration."""
        model_type = os.environ.get('MODEL_TYPE', 'ollama')
        logger.info(f"Initializing {agent_name} agent with {model_type} model...")
        
        # Get agent config with defaults
        agent_config = self.agents_config.get(agent_name, {})
        
        try:
            # Set up LLM based on model type
            if model_type == 'gemini':
                llm = LLM(
                    model="gemini-2.5-pro",
                    provider="google",
                    api_key=os.environ.get('GOOGLE_API_KEY'),
                    temperature=0.3,
                    max_tokens=2000
                )
            else:  # Default to Ollama
                llm_config = agent_config.get('llm', {})
                if isinstance(llm_config, str):
                    llm = LLM(
                        model=llm_config,
                        base_url="http://localhost:11434",
                        temperature=0.3,
                        max_tokens=2000
                    )
                else:
                    llm = LLM(**llm_config)
            
            # Create the agent
            agent = Agent(
                role=agent_config.get('role', ''),
                goal=agent_config.get('goal', ''),
                backstory=agent_config.get('backstory', ''),
                llm=llm,
                verbose=True,
                allow_delegation=False,
                tools=[JoltTransformTool()]
            )
            logger.info(f"{agent_name} agent initialized successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to initialize {agent_name} agent: {str(e)}", exc_info=True)
            raise

    def _read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Helper method to read a JSON file."""
        try:
            with open(file_path, 'r') as f:
                logger.debug(f"Reading JSON file: {file_path}")
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {file_path}", exc_info=True)
            raise ValueError(f"Invalid JSON in file: {file_path}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}", exc_info=True)
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error reading JSON file: {file_path}", exc_info=True)
            raise ValueError(f"Error reading JSON file: {file_path}")

    @task
    def generate_jolt_spec(self) -> Task:
        """Create a task to generate JOLT specification using configuration from tasks.yaml."""
        logger.info("Creating JOLT spec generation task...")
        
        try:
            # Get the task configuration
            task_config = self.tasks_config['generate_jolt_spec'].copy()
            
            # Create the task with configuration from tasks.yaml
            task = Task(
                description=task_config.get('description', ''),
                expected_output=task_config.get('expected_output', 'A complete JOLT specification'),
                agent=self.ocsf_jolt_parser(),
                output_file=task_config.get('output_file', 'jolt_spec.json'),
                output_format='json'
            )
            
            # Add a method to process inputs when the task is executed
            def process_inputs(self, task_inputs: dict[str, Any]) -> dict[str, Any]:
                # Get input file paths from the inputs
                input_file = task_inputs.get('input_file')
                output_template_file = task_inputs.get('output_template')
                field_mappings = task_inputs.get('field_mappings', {})
                
                # Validate inputs
                if not input_file or not output_template_file:
                    raise ValueError("Both input_file and output_template must be provided")
                    
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"Input file not found: {input_file}")
                    
                if not os.path.exists(output_template_file):
                    raise FileNotFoundError(f"Output template file not found: {output_template_file}")
                
                # Read the input files
                logger.debug(f"Reading input file: {input_file}")
                input_data = self._read_json_file(input_file)
                logger.debug(f"Reading output template file: {output_template_file}")
                output_template = self._read_json_file(output_template_file)
                
                # Update the task description with actual file contents
                description = (
                    f"Input OCSF Log (first 5 records):\n{json.dumps(input_data[:5] if isinstance(input_data, list) else input_data, indent=2)}\n\n"
                    f"Expected Output Template:\n{json.dumps(output_template, indent=2)}\n\n"
                )
                
                if field_mappings:
                    description += f"Field Mappings to Consider:\n{json.dumps(field_mappings, indent=2)}\n\n"
                
                self.description = description + task_config.get('description', '')
                
                # Return the processed inputs
                return {
                    'input_data': input_data,
                    'output_template': output_template,
                    'field_mappings': field_mappings
                }
            
            # Attach the callback to the task
            task._process_inputs = process_inputs.__get__(task, Task)
            
            logger.info("Task created successfully")
            return task
            
        except KeyError as e:
            logger.error(f"Missing required configuration: {str(e)}", exc_info=True)
            raise ValueError(f"Missing required configuration: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}", exc_info=True)
            raise Exception(f"Error creating task: {str(e)}")

    @task
    def validate_jolt_spec(self) -> Task:
        """Create a task to validate the generated JOLT spec."""
        logger.info("Creating JOLT spec validation task...")
        
        try:
            # Get task config with defaults
            task_config = self.tasks_config.get('validate_jolt_spec', {})
            
            def process_inputs(self_task, task_inputs: dict[str, Any]) -> dict[str, Any]:
                """Process inputs for the validation task."""
                input_file = task_inputs.get('input_file')
                output_template = task_inputs.get('output_template')
                # Default to 'jolt_spec.json' if not provided
                generated_spec_file = task_inputs.get('output_file', 'jolt_spec.json')
                
                # Log the inputs for debugging
                logger.debug(f"Validation task inputs - input_file: {input_file}")
                logger.debug(f"Validation task inputs - output_template: {output_template}")
                logger.debug(f"Validation task inputs - generated_spec_file: {generated_spec_file}")
                
                # Validate inputs
                if not all([input_file, output_template, generated_spec_file]):
                    raise ValueError("input_file, output_template, and generated_spec_file must be provided")
                
                # Read files
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                    with open(output_template, 'r', encoding='utf-8') as f:
                        expected_output = json.load(f)
                    with open(generated_spec_file, 'r', encoding='utf-8') as f:
                        jolt_spec = json.load(f)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Required file not found: {e.filename}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file: {e.doc}")
                
                # Return inputs for the task
                return {
                    'input_data': input_data,
                    'expected_output': expected_output,
                    'jolt_spec': jolt_spec,
                    'generated_spec_file': generated_spec_file
                }
            
            # Define the expected output format
            expected_output = """A JSON report with the following structure:
            {
                "success": boolean,  # Whether the validation passed
                "differences": [
                    {
                        "path": "path.to.field",  # JSON path to the field with mismatch
                        "actual": "actual_value",  # The actual value from the transformed output
                        "expected": "expected_value"  # The expected value from the template
                    }
                ]
            }
            """

            # Get the raw description
            raw_description = task_config.get('description', '')
            
            # First, escape all curly braces by doubling them
            escaped_description = raw_description.replace('{', '{{').replace('}', '}}')
            
            # Then unescape the actual template variables we want to keep
            description = (
                escaped_description
                .replace('{{input_file}}', '{input_file}')
                .replace('{{output_template}}', '{output_template}')
                .replace('{{output_file}}', '{output_file}')
                .replace('{{generated_spec_file}}', '{generated_spec_file}')
            )
            
            # Create the task with proper variable passing
            task = Task(
                description=description,
                expected_output=expected_output,
                agent=self.jolt_spec_validator(),
                output_file=task_config.get('output_file', 'validation_report.json'),
                context=[self.generate_jolt_spec()],  # Depends on generation task
                config={
                    'input_file': '{input_file}',
                    'output_template': '{output_template}',
                    'output_file': '{output_file}',
                    'generated_spec_file': '{generated_spec_file}'
                }
            )
            
            # Attach the callback to the task
            task._process_inputs = process_inputs.__get__(task, Task)
            
            logger.info("Validation task created successfully")
            return task
            
        except Exception as e:
            logger.error(f"Error creating validation task: {str(e)}", exc_info=True)
            raise Exception(f"Error creating validation task: {str(e)}")

    @crew
    def crew(self) -> Crew:
        """Create the JoltOcsfParserAi crew"""
        logger.info("Creating JoltOcsfParserAi crew...")
        
        try:
            # Initialize agents first
            logger.info("Initializing agents...")
            agent_gen = self.ocsf_jolt_parser()
            agent_val = self.jolt_spec_validator()
            
            # Set up tasks with proper variable passing
            logger.info("Preparing tasks...")
            
            # Get the output file for the generated spec from the config
            gen_task_config = self.tasks_config.get('generate_jolt_spec', {})
            output_file = gen_task_config.get('output_file', 'jolt_spec.json')
            
            # Create tasks with proper variable passing
            gen_task = self.generate_jolt_spec()
            
            # Pass the output_file from generate task to validate task
            val_task = self.validate_jolt_spec()
            
            # Set up task dependencies and variable passing
            tasks = [gen_task, val_task]
            
            logger.info("Creating crew with agents and tasks...")
            crew = Crew(
                agents=[agent_gen, agent_val],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )
            
            logger.info("Crew created successfully with generation and validation tasks")
            return crew
            
        except Exception as e:
            logger.error(f"Failed to create crew: {str(e)}", exc_info=True)
            raise
