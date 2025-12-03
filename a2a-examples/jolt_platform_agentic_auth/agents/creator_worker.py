#!/usr/bin/env python3
"""
Creator Agent Worker
Runs the CrewAI Creator Agent in a standalone process/container.
Connects to Kafka to receive START_WORKFLOW messages.
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jolt_platform.messaging import get_message_bus
from jolt_platform.agent_wrappers import CreatorAgentWrapper
from agents.crewai_jolt_agent import JoltSpecificationCreator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreatorWorker")

def main():
    logger.info("🚀 Starting Creator Agent Worker...")
    
    # Initialize Message Bus (Kafka or InMemory)
    bus = get_message_bus()
    
    # Initialize Agent
    agent = JoltSpecificationCreator()
    
    # Initialize Wrapper
    wrapper = CreatorAgentWrapper(agent, bus, "creator")
    # setup_subscriptions is called in __init__
    
    logger.info("✅ Creator Agent initialized and subscribed to START_WORKFLOW")
    
    # Start consuming messages (blocking)
    bus.start_consuming()

if __name__ == "__main__":
    main()
