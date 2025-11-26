#!/usr/bin/env python
import sys
import os
import json
import logging
import argparse
import re
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)
logger = logging.getLogger(__name__)

from jolt_ocsf_parser_ai.crew import JoltOcsfParserAi

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def validate_file(file_path: str) -> str:
    """Validate that the file exists and is a valid JSON file."""
    logger.info(f"Validating file: {file_path}")
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            logger.debug(f"File content (first 200 chars): {content[:200]}...")
            
            # Parse and validate JSON
            try:
                json_content = json.loads(content)
                logger.info(f"Successfully parsed JSON from {file_path}")
                logger.debug(f"JSON content type: {type(json_content).__name__}")
                
                # Log detailed structure for debugging
                if isinstance(json_content, dict):
                    logger.debug(f"JSON has {len(json_content)} top-level keys")
                    logger.debug(f"JSON keys: {list(json_content.keys())}")
                    # Log first few items of each key to understand structure
                    for i, (key, value) in enumerate(json_content.items()):
                        if i < 3:  # Only log first 3 items to avoid too much output
                            logger.debug(f"  {key}: type={type(value).__name__}, value={str(value)[:100]}")
                        else:
                            logger.debug(f"  ... and {len(json_content) - 3} more items")
                            break
                elif isinstance(json_content, list):
                    logger.debug(f"JSON is a list with {len(json_content)} items")
                    if json_content:
                        logger.debug(f"First item type: {type(json_content[0]).__name__}")
                        if isinstance(json_content[0], dict) and json_content[0]:
                            logger.debug(f"First item keys: {list(json_content[0].keys())}")
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse JSON from {file_path}")
                logger.error(f"Error position: {je.pos}, line {je.lineno}, column {je.colno}")
                # Show problematic line
                lines = content.splitlines()
                if je.lineno - 1 < len(lines):
                    problem_line = lines[je.lineno - 1]
                    logger.error(f"Problematic line {je.lineno}: {problem_line}")
                    logger.error(" " * (je.colno - 1) + "^-- Error here")
                raise
                
        return os.path.abspath(file_path)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in file {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

def load_mappings(mappings_file: Optional[str]) -> Dict[str, Any]:
    """Load field mappings if provided."""
    if not mappings_file:
        logger.info("No mappings file provided, using empty mappings")
        return {}
    try:
        logger.info(f"Loading mappings from: {mappings_file}")
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
            logger.debug(f"Loaded mappings: {json.dumps(mappings, indent=2)}")
            return mappings
    except Exception as e:
        error_msg = f"Error loading mappings file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

def clean_json_output(output_str: str) -> str:
    """
    Cleans the output string to extract valid JSON.
    Removes markdown code blocks and extra whitespace.
    """
    # Remove markdown code blocks
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1)
    return output_str.strip()

def run():
    """Run the OCSF to JOLT conversion."""
    logger.info("Starting OCSF to JOLT conversion")
    
    parser = argparse.ArgumentParser(description='Generate JOLT specification from OCSF logs')
    parser.add_argument('input_file', help='Path to the input OCSF JSON log file')
    parser.add_argument('output_template', help='Path to the expected output JSON template')
    parser.add_argument('--mappings', help='Path to field mappings JSON file (optional)')
    parser.add_argument('--output', '-o', default='jolt_spec.json', 
                       help='Output file for the JOLT specification (default: jolt_spec.json)')
    parser.add_argument('--model', choices=['ollama', 'gemini'], default='ollama',
                      help='Select the model to use: ollama (default) or gemini')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set log level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Set the model type in environment
    os.environ['MODEL_TYPE'] = args.model
    logger.info(f"Using model: {args.model}")
    
    try:
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output template: {args.output_template}")
        logger.info(f"Mappings file: {args.mappings}")
        logger.info(f"Output file: {args.output}")
        
        # Validate input files
        logger.info("Validating input files...")
        input_file = validate_file(args.input_file)
        output_template = validate_file(args.output_template)
        
        # Load mappings if provided
        field_mappings = {}
        if args.mappings:
            field_mappings = load_mappings(args.mappings)
        
        # Prepare inputs for the crew
        inputs = {
            'input_file': input_file,
            'output_template': output_template,
            'field_mappings': field_mappings,
            'model': args.model,
            'generated_spec_file': args.output,  # Add this line to pass the output file path to the validation task
            'output_file': args.output  # This is used by the generation task
        }
        logger.debug(f"Crew inputs: {json.dumps(inputs, indent=2)}")
        
        # Run the crew
        print(f"\nGenerating JOLT specification from:")
        print(f"  Input: {input_file}")
        print(f"  Template: {output_template}")
        if field_mappings:
            print(f"  Using field mappings from: {args.mappings}")
        
        logger.info("Initializing JoltOcsfParserAi crew...")
        try:
            # Create the crew instance
            logger.debug("Creating JoltOcsfParserAi instance...")
            jolt_crew = JoltOcsfParserAi()
            logger.debug("JoltOcsfParserAi instance created")
            
            # Get the crew
            logger.debug("Getting crew instance...")
            crew = jolt_crew.crew()
            logger.debug("Crew instance created")
            
            # Log the inputs being passed to kickoff
            logger.info("Crew initialized, preparing to kick off the task...")
            logger.debug(f"Inputs to crew.kickoff(): {json.dumps(inputs, default=str, indent=2)}")
            
            # Add type checking for inputs
            for key, value in inputs.items():
                logger.debug(f"Input '{key}' type: {type(value).__name__}")
                if isinstance(value, dict):
                    logger.debug(f"Input '{key}' keys: {list(value.keys())}")
            
            # Run the crew
            logger.info("Kicking off the task...")
            result = crew.kickoff(inputs=inputs)
            logger.info("Task execution completed")
        except Exception as e:
            logger.error("Error during crew initialization or execution", exc_info=True)
            raise
        logger.info("Task completed successfully")
        
        # Handle the output
        # The result of kickoff is the output of the last task (validation task)
        # We need to find the output of the generation task to save as the Jolt spec
        
        jolt_spec_output = None
        
        # Check if we have access to individual task outputs
        if hasattr(result, 'tasks_output') and result.tasks_output:
            logger.info(f"Found {len(result.tasks_output)} task outputs")
            # Assuming the first task is the generation task as defined in crew.py
            # tasks = [gen_task, val_task]
            if len(result.tasks_output) >= 1:
                jolt_spec_output = result.tasks_output[0].raw
                logger.info("Retrieved output from the first task (generation task)")
        else:
            # Fallback: if we can't access tasks_output, we might be in a mode where
            # only the final result is available. But since we know the final result
            # is the validation report, this is problematic.
            # However, if the user ran ONLY the generation task (e.g. via some other means),
            # then result.raw_output might be the spec.
            # For now, let's warn if we can't find the spec.
            logger.warning("Could not access individual task outputs. Checking if final result looks like a spec.")
            result_data = result.raw if hasattr(result, 'raw') else str(result)
            if isinstance(result_data, str) and "operation" in result_data and "shift" in result_data:
                 jolt_spec_output = result_data
            else:
                 logger.error("Could not locate Jolt spec in task outputs.")

        if jolt_spec_output:
            # Clean and parse the output
            final_data = jolt_spec_output
            logger.info("[INFO] Checking if jolt_spec_output is a string")
            if isinstance(jolt_spec_output, str):
                logger.info("[INFO] jolt_spec_output is a string")
                cleaned_data = clean_json_output(jolt_spec_output)
                try:
                    final_data = json.loads(cleaned_data)
                    logger.info("[INFO] Successfully cleaned and parsed JSON output")
                except json.JSONDecodeError:
                    logger.warning("[WARNING] Could not parse cleaned output as JSON, saving as raw string")
                    final_data = cleaned_data
            
            # Save the output
            output_path = Path(args.output).absolute()
            with open(output_path, 'w') as f:
                json.dump(final_data, f, indent=2, default=str)
            
            print(f"\n✅ JOLT specification successfully generated and saved to: {output_path}")
            logger.info(f"[SUCCESS] Output saved to {output_path}")
        else:
            logger.error("Failed to extract Jolt specification from crew output.")
            print("\n❌ Error: Failed to extract Jolt specification from crew output.", file=sys.stderr)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"An error occurred: {str(e)}\n{error_trace}")
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        print("\nDetailed error information has been logged to debug.log", file=sys.stderr)
        print("\nCommon causes for this error:", file=sys.stderr)
        print("1. The Ollama model may not be properly loaded or responding", file=sys.stderr)
        print("2. There might be an issue with the input JSON structure", file=sys.stderr)
        print("3. The field mappings might contain invalid data", file=sys.stderr)
        print("\nTry running with --debug for more detailed information", file=sys.stderr)
        sys.exit(1)

def replay():
    """Replay the crew execution from a specific task."""
    try:
        if len(sys.argv) < 2:
            print("Usage: python -m jolt_ocsf_parser_ai replay <task_id>")
            sys.exit(1)
            
        JoltOcsfParserAi().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        print(f"Error replaying task: {str(e)}", file=sys.stderr)
        sys.exit(1)


def test():
    """Test the OCSF to JOLT conversion with sample data."""
    try:
        # Create a sample test case
        sample_input = {
            "activity_id": 1,
            "activity_name": "Process Start",
            "category_uid": 1,
            "class_uid": 1,
            "cloud": {
                "provider": "AWS",
                "region": "us-west-2"
            },
            "metadata": {
                "product": {
                    "name": "Test Product"
                },
                "version": "1.0.0"
            },
            "severity": "Low",
            "severity_id": 1,
            "time": "2023-01-01T00:00:00Z",
            "type_uid": 1
        }
        
        sample_template = {
            "event": {
                "id": "",
                "name": "",
                "severity": "",
                "timestamp": "",
                "cloud": {
                    "provider": "",
                    "region": ""
                }
            },
            "metadata": {
                "product": "",
                "version": ""
            }
        }
        
        # Save sample files
        with open('sample_ocsf.json', 'w') as f:
            json.dump(sample_input, f, indent=2)
        
        with open('sample_template.json', 'w') as f:
            json.dump(sample_template, f, indent=2)
        
        print("Running test with sample data...")
        print("Sample OCSF input saved to: sample_ocsf.json")
        print("Sample template saved to: sample_template.json")
        
        # Run the conversion
        inputs = {
            'input_file': 'sample_ocsf.json',
            'output_template': 'sample_template.json'
        }
        
        result = JoltOcsfParserAi().crew().kickoff(inputs=inputs)
        
        # Save the result
        with open('sample_jolt_spec.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("\nTest completed successfully!")
        print("JOLT specification saved to: sample_jolt_spec.json")
        
        # Clean up sample files
        os.remove('sample_ocsf.json')
        os.remove('sample_template.json')
        
    except Exception as e:
        print(f"Test failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_with_trigger():
    """Run the crew with trigger payload from command line or file."""
    parser = argparse.ArgumentParser(description='Run OCSF to JOLT conversion with trigger payload')
    parser.add_argument('payload_file', nargs='?', 
                       help='Path to JSON file containing trigger payload')
    
    # Parse just the known arguments to avoid interfering with the rest of the script
    args, _ = parser.parse_known_args()
    
    try:
        if args.payload_file:
            # Load payload from file
            with open(args.payload_file, 'r') as f:
                trigger_payload = json.load(f)
        else:
            # Try to read from command line argument
            if len(sys.argv) > 2:
                trigger_payload = json.loads(sys.argv[2])
            else:
                raise ValueError("No payload provided. Please provide a JSON file or JSON string.")
        
        # Run the crew with the trigger payload
        print("Running with trigger payload:", json.dumps(trigger_payload, indent=2))
        result = JoltOcsfParserAi().crew().kickoff(inputs=trigger_payload)
        
        # Print the result
        print("\nJOLT specification generated successfully!")
        print(json.dumps(result, indent=2))
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON payload: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running with trigger: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Use command line arguments to determine which function to run
    if len(sys.argv) > 1 and sys.argv[1] == 'replay':
        replay()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'run-with-trigger':
        run_with_trigger()
    else:
        # Default to run mode with file arguments
        run()
