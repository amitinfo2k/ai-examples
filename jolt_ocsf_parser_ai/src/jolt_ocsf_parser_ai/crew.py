import json
import os
import logging
from typing import Dict, Any, List, Optional

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

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
        # Get model type from environment variable or use 'ollama' as default
        model_type = os.environ.get('MODEL_TYPE', 'ollama')
        logger.info(f"Initializing OCSF to JOLT parser agent with {model_type} model...")
        agent_config = self.agents_config['ocsf_jolt_parser'].copy()
        
        try:
            if model_type == 'gemini':
                # For Google's Gemini, we explicitly set the provider to 'google'
                llm = LLM(
                    model="gemini-2.5-pro",  # Updated to the latest Gemini Pro model
                    provider="google",
                    api_key=os.environ.get('GOOGLE_API_KEY'),
                    temperature=0.7,
                    max_tokens=4000
                )
            else:  # Default to Ollama
                llm_config = agent_config.get('llm', {})
                if isinstance(llm_config, str):
                    model = llm_config
                    llm = LLM(
                        model=model,
                        base_url="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=4000
                    )
                else:
                    llm = LLM(**llm_config)
            
            # Create the agent using configuration from agents.yaml
            agent = Agent(
                role=agent_config.get('role', ''),
                goal=agent_config.get('goal', ''),
                backstory=agent_config.get('backstory', ''),
                llm=llm,
                verbose=True,
                allow_delegation=False,
                tools=[]
            )
            logger.info("Agent initialized successfully")
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}", exc_info=True)
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
    def generate_jolt_spec_task(self) -> Task:
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

    @crew
    def crew(self) -> Crew:
        """Create the JoltOcsfParserAi crew"""
        logger.info("Creating JoltOcsfParserAi crew...")
        
        try:
            # Set up the main task
            logger.info("Generating JOLT spec task...")
            self.tasks = [self.generate_jolt_spec_task()]  # type: ignore
            
            logger.info("Initializing agent...")
            agent = self.ocsf_jolt_parser()
            
            logger.info("Creating crew with agent and tasks...")
            crew = Crew(
                agents=[agent],
                tasks=self.tasks,  # type: ignore
                process=Process.sequential,
                verbose=True,
            )
            
            logger.info("Crew created successfully")
            return crew
            
        except Exception as e:
            logger.error(f"Failed to create crew: {str(e)}", exc_info=True)
            raise
