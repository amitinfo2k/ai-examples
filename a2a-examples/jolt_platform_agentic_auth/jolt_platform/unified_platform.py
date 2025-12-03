"""
Unified Platform for JOLT Specification Creation and Validation
This platform orchestrates both CrewAI and LangChain agents to provide
a complete JOLT specification workflow.
"""

import sys
import os

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.crewai_jolt_agent import JoltSpecificationCreator
from agents.langchain_validation_agent import JoltValidator
from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path


class JoltPlatform:
    """
    Unified platform for JOLT specification creation and validation.
    Designed with a modular architecture to allow independent agent registration.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the JOLT platform.
        
        Args:
            output_dir: Directory to save output files
        """
        print("🚀 Initializing JOLT Platform...")
        print("=" * 60)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Agent Registry
        self.agents = {
            "creator": None,
            "validator": None
        }
        
        print("✨ Platform initialized! (No agents loaded yet)")
        print("👉 Use 'register_agent()' or 'load_default_agents()' to setup agents.\n")

    def register_agent(self, role: str, agent: Any):
        """
        Register an agent for a specific role.
        
        Args:
            role: 'creator' or 'validator'
            agent: The initialized agent instance
        """
        if role not in self.agents:
            print(f"⚠️ Warning: Unknown role '{role}'. Registering anyway.")
        
        self.agents[role] = agent
        print(f"✅ Registered agent for role: {role}")

    def load_default_agents(self, model_name: str = None):
        """
        Load the default CrewAI and LangChain agents (for backward compatibility).
        
        Args:
            model_name: Gemini model to use
        """
        model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        print(f"📦 Loading default agents with model: {model_name}...")
        
        # Lazy import to avoid tight coupling at module level
        from agents.crewai_jolt_agent import JoltSpecificationCreator
        from agents.langchain_validation_agent import JoltValidator
        
        # Initialize and register
        self.register_agent("creator", JoltSpecificationCreator(model_name=model_name))
        self.register_agent("validator", JoltValidator(model_name=model_name))
        print("✅ Default agents loaded successfully!\n")

    @property
    def spec_creator(self):
        """Access the creator agent."""
        if not self.agents["creator"]:
            raise ValueError("No creator agent registered! Use register_agent('creator', agent)")
        return self.agents["creator"]

    @property
    def validator(self):
        """Access the validator agent."""
        if not self.agents["validator"]:
            raise ValueError("No validator agent registered! Use register_agent('validator', agent)")
        return self.agents["validator"]
    
    def create_and_validate(
        self,
        input_json: Dict[str, Any],
        expected_output: Dict[str, Any],
        save_outputs: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow: Create JOLT spec and validate it.
        
        Args:
            input_json: Input JSON structure
            expected_output: Expected output JSON structure
            save_outputs: Whether to save intermediate outputs
            
        Returns:
            Dictionary containing jolt_spec, validation_report, and status
        """
        print("\n" + "=" * 60)
        print("🔄 Starting JOLT Spec Creation and Validation Workflow")
        print("=" * 60)
        
        # Step 1: Create JOLT specification using CrewAI
        print("\n📝 STEP 1: Creating JOLT Specification with CrewAI Agent...")
        print("-" * 60)
        
        try:
            jolt_spec = self.spec_creator.create_jolt_spec(input_json, expected_output)
            print("✅ JOLT Specification created successfully!")
            
            if save_outputs:
                spec_file = self.output_dir / f"jolt_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.spec_creator.save_jolt_spec(jolt_spec, str(spec_file))
        except Exception as e:
            print(f"❌ Error creating JOLT specification: {e}")
            return {
                "status": "failed",
                "stage": "creation",
                "error": str(e)
            }
        
        # Step 2: Validate JOLT specification using LangChain
        print("\n✅ STEP 2: Validating JOLT Specification with LangChain Agent...")
        print("-" * 60)
        
        try:
            validation_report = self.validator.validate_jolt_spec(
                jolt_spec,
                input_json,
                expected_output
            )
            print("✅ Validation completed successfully!")
            
            if save_outputs:
                report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.validator.save_validation_report(validation_report, str(report_file))
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return {
                "status": "failed",
                "stage": "validation",
                "jolt_spec": jolt_spec,
                "error": str(e)
            }
        
        # Compile results
        print("\n" + "=" * 60)
        print("🎉 Workflow Complete!")
        print("=" * 60)
        
        result = {
            "status": "success",
            "jolt_spec": jolt_spec,
            "validation_report": validation_report,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def create_spec_only(
        self,
        input_json: Dict[str, Any],
        expected_output: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create JOLT specification only (skip validation).
        
        Args:
            input_json: Input JSON structure
            expected_output: Expected output JSON structure
            output_file: Optional path to save the specification
            
        Returns:
            JOLT specification
        """
        print("\n📝 Creating JOLT Specification (CrewAI Agent)...")
        
        jolt_spec = self.spec_creator.create_jolt_spec(input_json, expected_output)
        
        if output_file:
            self.spec_creator.save_jolt_spec(jolt_spec, output_file)
        
        print("✅ JOLT Specification created!")
        return jolt_spec
    
    def validate_spec_only(
        self,
        jolt_spec: Dict[str, Any],
        input_json: Dict[str, Any],
        expected_output: Dict[str, Any],
        report_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate an existing JOLT specification (skip creation).
        
        Args:
            jolt_spec: JOLT specification to validate
            input_json: Input JSON structure
            expected_output: Expected output JSON structure
            report_file: Optional path to save the validation report
            
        Returns:
            Validation report
        """
        print("\n✅ Validating JOLT Specification (LangChain Agent)...")
        
        validation_report = self.validator.validate_jolt_spec(
            jolt_spec,
            input_json,
            expected_output
        )
        
        if report_file:
            self.validator.save_validation_report(validation_report, report_file)
        
        print("✅ Validation complete!")
        return validation_report
    
    def enable_a2a_mode(self):
        """Enable Agent-to-Agent communication mode."""
        from jolt_platform.messaging import get_message_bus, Message
        from jolt_platform.agent_wrappers import CreatorAgentWrapper, ValidatorAgentWrapper
        
        # Use factory to get appropriate bus (Kafka or InMemory)
        self.bus = get_message_bus()
        print(f"📡 A2A Messaging Bus enabled ({type(self.bus).__name__})")
        
        # Only wrap local agents if they exist and we are using InMemory bus
        # In distributed mode (Kafka), agents run in separate containers
        if "InMemory" in type(self.bus).__name__:
            if self.agents["creator"]:
                self.creator_wrapper = CreatorAgentWrapper(self.agents["creator"], self.bus, "creator")
                # setup_subscriptions is called in __init__
                print("   - Wrapped Creator Agent")
                
            if self.agents["validator"]:
                self.validator_wrapper = ValidatorAgentWrapper(self.agents["validator"], self.bus, "validator")
                # setup_subscriptions is called in __init__
                print("   - Wrapped Validator Agent")
        else:
            print("   - Distributed Mode: Agents expected to be running externally")

    def run_a2a_workflow(self, input_json: Dict[str, Any], expected_output: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the workflow using Agent-to-Agent messaging protocol.
        Instead of orchestrating calls, we publish a start message and let agents react.
        """
        if not hasattr(self, 'bus'):
            self.enable_a2a_mode()
            
        print("\n" + "=" * 60)
        print(f"📡 Starting A2A Workflow (Event-Driven) [Job ID: {job_id}]")
        print("=" * 60)
        
        # Start workflow
        from jolt_platform.messaging import Message
        payload = {
            "input_json": input_json, 
            "expected_output": expected_output
        }
        if job_id:
            payload["job_id"] = job_id
            
        self.bus.publish(Message(
            type="START_WORKFLOW",
            payload=payload,
            sender="platform"
        ))
        
        return {"status": "initiated", "job_id": job_id, "message": "Workflow started successfully"}
        


    def process_from_files(
        self,
        input_file: str,
        output_file: str,
        jolt_spec_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process JOLT workflow from files.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to expected output JSON file
            jolt_spec_file: Optional path to existing JOLT spec (if None, will create new)
            
        Returns:
            Workflow results
        """
        # Load JSON files
        with open(input_file, 'r') as f:
            input_json = json.load(f)
        
        with open(output_file, 'r') as f:
            expected_output = json.load(f)
        
        # If JOLT spec provided, validate only; otherwise, full workflow
        if jolt_spec_file:
            with open(jolt_spec_file, 'r') as f:
                jolt_spec = json.load(f)
            report = self.validate_spec_only(jolt_spec, input_json, expected_output)
            return {
                "status": "success",
                "mode": "validation_only",
                "validation_report": report
            }
        else:
            return self.create_and_validate(input_json, expected_output)


def main():
    """Main function demonstrating platform usage."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Mode selection
    print("\n" + "=" * 60)
    print("JOLT Multi-Agent Platform Demo")
    print("=" * 60)
    print("\n🔄 Workflow Mode Selection")
    print("=" * 60)
    print("1. Traditional Mode (Platform orchestrates agents)")
    print("2. A2A Mode (Event-driven, agents communicate via messages)")
    print("=" * 60)
    
    mode_choice = input("\nSelect mode (1 or 2, default=1): ").strip()
    use_a2a = mode_choice == "2"
    
    # Initialize platform
    platform = JoltPlatform(output_dir="./output")
    platform.load_default_agents()
    
    if use_a2a:
        platform.enable_a2a_mode()
        print("\n📡 A2A Mode Enabled - Agents will communicate via Message Bus")
    else:
        print("\n🔧 Traditional Mode - Platform will orchestrate agents")
    
    # Example 1: Simple transformation
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Simple Field Mapping")
    print("=" * 60)
    
    input_json_1 = {
        "user": {
            "firstName": "John",
            "lastName": "Doe",
            "email": "john.doe@example.com"
        },
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    expected_output_1 = {
        "fullName": "John Doe",
        "contact": {
            "email": "john.doe@example.com"
        },
        "eventtime": "2024-01-01T12:00:00Z"
    }
    
    if use_a2a:
        result_1 = platform.run_a2a_workflow(input_json_1, expected_output_1)
    else:
        result_1 = platform.create_and_validate(input_json_1, expected_output_1)
    
    print(f"\nResult Status: {result_1.get('status', 'success')}")
    if result_1.get('status') == 'success' or 'jolt_spec' in result_1:
        print("\nGenerated JOLT Spec:")
        print(json.dumps(result_1['jolt_spec'], indent=2))
    
    # Example 2: More complex transformation
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: Complex Nested Structure")
    print("=" * 60)
    
    input_json_2 = {
        "order": {
            "id": "ORD-123",
            "customer": {
                "name": "Alice Smith",
                "email": "alice@example.com"
            },
            "items": [
                {"product": "Laptop", "price": 999.99},
                {"product": "Mouse", "price": 29.99}
            ]
        }
    }
    
    expected_output_2 = {
        "orderId": "ORD-123",
        "customerInfo": {
            "customerName": "Alice Smith",
            "contactEmail": "alice@example.com"
        },
        "orderItems": [
            {"productName": "Laptop", "amount": 999.99},
            {"productName": "Mouse", "amount": 29.99}
        ]
    }
    
    if use_a2a:
        result_2 = platform.run_a2a_workflow(input_json_2, expected_output_2)
    else:
        result_2 = platform.create_and_validate(input_json_2, expected_output_2)
    
    print(f"\nResult Status: {result_2.get('status', 'success')}")
    if result_2.get('status') == 'success' or 'jolt_spec' in result_2:
        print("\nGenerated JOLT Spec:")
        print(json.dumps(result_2['jolt_spec'], indent=2))


if __name__ == "__main__":
    main()
