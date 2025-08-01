#!/usr/bin/env python3

import asyncio
import logging
from k8s_debug.common.utils import get_mcp_server_diag_config
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_connection():
    """Test the MCP server connection."""
    try:
        config = get_mcp_server_diag_config()
        logger.info(f'Testing MCP server connection to: {config.url}')
        
        tools = await MCPToolset(
            connection_params=SseServerParams(url=config.url)
        ).get_tools()
        
        logger.info(f'Successfully connected! Found {len(tools)} tools:')
        for tool in tools:
            logger.info(f'  - {tool.name}')
            
        return True
    except Exception as e:
        logger.error(f'Failed to connect to MCP server: {e}')
        return False

if __name__ == "__main__":
    asyncio.run(test_mcp_connection()) 