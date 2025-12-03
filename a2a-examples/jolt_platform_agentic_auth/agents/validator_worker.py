#!/usr/bin/env python3
"""
Validator Agent Worker
Runs the LangChain Validator Agent in a standalone process/container.
Connects to Kafka to receive SPEC_CREATED messages.
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jolt_platform.messaging import get_message_bus
from jolt_platform.agent_wrappers import ValidatorAgentWrapper
from agents.langchain_validation_agent import JoltValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ValidatorWorker")

def main():
    logger.info("🚀 Starting Validator Agent Worker...")
    
    # Initialize Message Bus (Kafka or InMemory)
    bus = get_message_bus()
    
    # Initialize Agent
    agent = JoltValidator()
    
    # Initialize Wrapper
    wrapper = ValidatorAgentWrapper(agent, bus, "validator")
    # setup_subscriptions is called in __init__
    
    logger.info("✅ Validator Agent initialized and subscribed to SPEC_CREATED")
    
    # Start consuming messages (blocking)
    bus.start_consuming()

if __name__ == "__main__":
    main()
