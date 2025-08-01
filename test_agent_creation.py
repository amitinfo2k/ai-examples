#!/usr/bin/env python3

import asyncio
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'a2a-examples/k8s_a2a_mcp/src'))

from k8s_debug.agents.k8s_diagnostic_agent import K8sDiagnosticAgent
from k8s_debug.common import prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_creation():
    """Test agent creation and initialization."""
    try:
        logger.info('Testing K8sDiagnosticAgent creation...')
        
        # Create the agent
        agent = K8sDiagnosticAgent(
            agent_name='K8sDiagnosticAgent',
            description='Get k8s diagnostic info',
            instructions=prompts.K8S_DIAGNOSTIC_INSTRUCTIONS,
        )
        
        logger.info(f'Agent created successfully: {agent}')
        logger.info(f'Agent name: {agent.agent_name}')
        logger.info(f'Agent description: {agent.description}')
        
        # Test agent initialization
        logger.info('Testing agent initialization...')
        await agent.init_agent()
        
        if agent.agent:
            logger.info('Agent initialized successfully!')
            return True
        else:
            logger.error('Agent initialization failed - agent is None')
            return False
            
    except Exception as e:
        logger.error(f'Error in test_agent_creation: {e}')
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_creation())
    if success:
        print("✅ Agent creation and initialization test passed!")
    else:
        print("❌ Agent creation and initialization test failed!")
        sys.exit(1) 