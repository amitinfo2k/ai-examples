# type: ignore

import json
import logging
import re

from collections.abc import AsyncIterable
from typing import Any

from k8s_debug.common.agent_runner import AgentRunner
from k8s_debug.common.base_agent import BaseAgent
from k8s_debug.common.utils import get_mcp_server_config, init_api_key
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.genai import types as genai_types


logger = logging.getLogger(__name__)


class K8sDiagnosticAgent(BaseAgent):
    """K8s Diagnostic Agent backed by ADK."""

    def __init__(self, agent_name: str, description: str, instructions: str):
        logger.info(f'Initializing K8sDiagnosticAgent with name: {agent_name}')
        try:
            init_api_key()
            logger.info('API key initialized successfully')

            super().__init__(
                agent_name=agent_name,
                description=description,
                content_types=['text', 'text/plain'],
            )

            logger.info(f'Successfully initialized {self.agent_name}')

            self.instructions = instructions
            self.agent = None
            logger.info(f'K8sDiagnosticAgent initialization complete for {self.agent_name}')
        except Exception as e:
            logger.error(f'Error initializing K8sDiagnosticAgent: {e}')
            raise

    async def init_agent(self):
        logger.info(f'Initializing {self.agent_name} metadata')
        try:
            config = get_mcp_server_config()
            logger.info(f'MCP Server url={config.url}')
            
            # Try to connect to MCP server
            tools = await MCPToolset(
                connection_params=SseServerParams(url=config.url)
            ).get_tools()

            logger.info(f'Successfully connected to MCP server and loaded {len(tools)} tools')
            for tool in tools:
                logger.info(f'Loaded tool: {tool.name}')
                
            generate_content_config = genai_types.GenerateContentConfig(
                temperature=0.0
            )
            self.agent = Agent(
                name=self.agent_name,
                instruction=self.instructions,
                model='gemini-2.0-flash',
                disallow_transfer_to_parent=True,
                disallow_transfer_to_peers=True,
                generate_content_config=generate_content_config,
                tools=tools,
            )
            self.runner = AgentRunner()
            logger.info(f'Successfully initialized agent {self.agent_name}')
        except Exception as e:
            logger.error(f'Failed to initialize agent {self.agent_name}: {e}')
            # Don't re-raise the exception, just log it and leave self.agent as None
            # The stream method will handle this case

    async def invoke(self, query, session_id) -> dict:
        logger.info(f'Running {self.agent_name} for session {session_id}')

        raise NotImplementedError('Please use the streraming function')

    async def stream(
        self, query, context_id, task_id
    ) -> AsyncIterable[dict[str, Any]]:
        logger.info(
            f'Running {self.agent_name} stream for session {context_id} {task_id} - {query}'
        )

        if not query:
            raise ValueError('Query cannot be empty')

        # Ensure agent is initialized
        if not self.agent:
            logger.info(f'Initializing agent {self.agent_name}')
            await self.init_agent()
            logger.info(f'Agent {self.agent_name} initialized successfully')
        
        if not self.agent:
            logger.error(f'Failed to initialize agent {self.agent_name}')
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'Error: Failed to initialize agent {self.agent_name}',
            }
            return

        try:
            async for chunk in self.runner.run_stream(
                self.agent, query, context_id
            ):
                logger.info(f'Received chunk {chunk}')
                if isinstance(chunk, dict) and chunk.get('type') == 'final_result':
                    response = chunk['response']
                    yield self.get_agent_response(response)
                else:
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': f'{self.agent_name}: Processing Request...',
                    }
        except Exception as e:
            logger.error(f'Error in agent stream: {e}')
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'Error: {str(e)}',
            }

    def format_response(self, chunk):
        patterns = [
            r'```\n(.*?)\n```',
            r'```json\s*(.*?)\s*```',
            r'```tool_outputs\s*(.*?)\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, chunk, re.DOTALL)
            if match:
                content = match.group(1)
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
        return chunk

    def get_agent_response(self, chunk):
        logger.info(f'Response Type {type(chunk)}')
        data = self.format_response(chunk)
        logger.info(f'Formatted Response {data}')
        try:
            if isinstance(data, dict):
                if 'status' in data and data['status'] == 'input_required':
                    return {
                        'response_type': 'text',
                        'is_task_complete': False,
                        'require_user_input': True,
                        'content': data['question'],
                    }
                return {
                    'response_type': 'data',
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': data,
                }
            return_type = 'data'
            try:
                data = json.loads(data)
                return_type = 'data'
            except Exception as json_e:
                logger.error(f'Json conversion error {json_e} - {data}')
                return_type = 'text'
            return {
                'response_type': return_type,
                'is_task_complete': True,
                'require_user_input': False,
                'content': data,
            }
        except Exception as e:
            logger.error(f'Error in get_agent_response: {e}')
            return {
                'response_type': 'text',
                'is_task_complete': True,
                'require_user_input': False,
                'content': 'Could not complete booking / task. Please try again.',
            }