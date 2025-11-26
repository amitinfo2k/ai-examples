from typing import Any, Dict
from jolt_platform.messaging import MessageBus, Message

class AgentWrapper:
    """Base class for agent wrappers."""
    def __init__(self, agent: Any, message_bus: MessageBus, name: str):
        self.agent = agent
        self.bus = message_bus
        self.name = name
        self.setup_subscriptions()

    def setup_subscriptions(self):
        """Override to subscribe to messages."""
        pass

    def publish(self, type: str, payload: Dict[str, Any]):
        """Helper to publish messages."""
        msg = Message(
            type=type,
            payload=payload,
            sender=self.name
        )
        self.bus.publish(msg)

class CreatorAgentWrapper(AgentWrapper):
    """Wraps JoltSpecificationCreator to communicate via MessageBus."""
    
    def setup_subscriptions(self):
        self.bus.subscribe("START_WORKFLOW", self.handle_start_workflow)

    def handle_start_workflow(self, message: Message):
        print(f"\n[CreatorAgent] Received task: {message.type}")
        payload = message.payload
        input_json = payload.get("input_json")
        expected_output = payload.get("expected_output")
        job_id = payload.get("job_id")  # Extract job_id

        if not input_json or not expected_output:
            print("[CreatorAgent] Error: Missing input_json or expected_output")
            return

        # Call the actual agent
        print("[CreatorAgent] Generating JOLT spec...")
        try:
            jolt_spec = self.agent.create_jolt_spec(input_json, expected_output)
            
            # Publish result with job_id
            publish_payload = {
                "jolt_spec": jolt_spec,
                "input_json": input_json,
                "expected_output": expected_output
            }
            if job_id:
                publish_payload["job_id"] = job_id  # Include job_id
                
            self.publish("SPEC_CREATED", publish_payload)
            print("[CreatorAgent] Published SPEC_CREATED")
        except Exception as e:
            print(f"[CreatorAgent] Error: {e}")
            error_payload = {"error": str(e), "stage": "creation"}
            if job_id:
                error_payload["job_id"] = job_id
            self.publish("WORKFLOW_ERROR", error_payload)

class ValidatorAgentWrapper(AgentWrapper):
    """Wraps JoltValidator to communicate via MessageBus."""
    
    def setup_subscriptions(self):
        self.bus.subscribe("SPEC_CREATED", self.handle_spec_created)

    def handle_spec_created(self, message: Message):
        print(f"\n[ValidatorAgent] Received task: {message.type}")
        payload = message.payload
        jolt_spec = payload.get("jolt_spec")
        input_json = payload.get("input_json")
        expected_output = payload.get("expected_output")
        job_id = payload.get("job_id")  # Extract job_id

        # Call the actual agent
        print("[ValidatorAgent] Validating spec...")
        try:
            validation_report = self.agent.validate_jolt_spec(jolt_spec, input_json, expected_output)
            
            # Publish result with job_id
            validation_payload = {
                "validation_report": validation_report,
                "jolt_spec": jolt_spec
            }
            if job_id:
                validation_payload["job_id"] = job_id
                
            self.publish("VALIDATION_COMPLETED", validation_payload)
            print("[ValidatorAgent] Published VALIDATION_COMPLETED")
            
            # Also publish a generic workflow complete message with job_id
            complete_payload = {
                "status": "success",
                "result": {
                    "jolt_spec": jolt_spec,
                    "validation_report": validation_report
                }
            }
            if job_id:
                complete_payload["job_id"] = job_id
                
            self.publish("WORKFLOW_COMPLETE", complete_payload)
        except Exception as e:
            print(f"[ValidatorAgent] Error: {e}")
            error_payload = {"error": str(e), "stage": "validation"}
            if job_id:
                error_payload["job_id"] = job_id
            self.publish("WORKFLOW_ERROR", error_payload)
