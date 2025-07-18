# type:ignore
import asyncio
import json
import os

from contextlib import asynccontextmanager

import click

from fastmcp.utilities.logging import get_logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ReadResourceResult


logger = get_logger(__name__)

env = {
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
}


@asynccontextmanager
async def init_session(host, port, transport):
    """Initializes and manages an MCP ClientSession based on the specified transport.

    This asynchronous context manager establishes a connection to an MCP server
    using either Server-Sent Events (SSE) or Standard I/O (STDIO) transport.
    It handles the setup and teardown of the connection and yields an active
    `ClientSession` object ready for communication.

    Args:
        host: The hostname or IP address of the MCP server (used for SSE).
        port: The port number of the MCP server (used for SSE).
        transport: The communication transport to use ('sse' or 'stdio').

    Yields:
        ClientSession: An initialized and ready-to-use MCP client session.

    Raises:
        ValueError: If an unsupported transport type is provided (implicitly,
                    as it won't match 'sse' or 'stdio').
        Exception: Other potential exceptions during client initialization or
                   session setup.
    """
    if transport == 'sse':
        url = f'http://{host}:{port}/sse'
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(
                read_stream=read_stream, write_stream=write_stream
            ) as session:
                logger.debug('SSE ClientSession created, initializing...')
                await session.initialize()
                logger.info('SSE ClientSession initialized successfully.')
                yield session
    elif transport == 'stdio':
        if not os.getenv('GOOGLE_API_KEY'):
            logger.error('GOOGLE_API_KEY is not set')
            raise ValueError('GOOGLE_API_KEY is not set')
        stdio_params = StdioServerParameters(
            command='uv',
            args=['run', 'a2a-mcp'],
            env=env,
        )
        async with stdio_client(stdio_params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
            ) as session:
                logger.debug('STDIO ClientSession created, initializing...')
                await session.initialize()
                logger.info('STDIO ClientSession initialized successfully.')
                yield session
    else:
        logger.error(f'Unsupported transport type: {transport}')
        raise ValueError(
            f"Unsupported transport type: {transport}. Must be 'sse' or 'stdio'."
        )


async def find_agent(session: ClientSession, query) -> CallToolResult:
    """Calls the 'find_agent' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'find_agent' tool.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'find_agent' tool with query: '{query[:50]}...'")
    return await session.call_tool(
        name='find_agent',
        arguments={
            'query': query,
        },
    )


async def find_resource(session: ClientSession, resource) -> ReadResourceResult:
    """Reads a resource from the connected MCP server.

    Args:
        session: The active ClientSession.
        resource: The URI of the resource to read (e.g., 'resource://agent_cards/list').

    Returns:
        The result of the resource read operation.
    """
    logger.info(f'Reading resource: {resource}')
    return await session.read_resource(resource)


async def list_pods(session: ClientSession) -> CallToolResult:
    """Calls the 'list_pods' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'list_pods' tool.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'list_pods' tool'")
    return await session.call_tool(
        name='list_pods',
        arguments={
            'namespace': 'default',
            'podname': 'test',
            'container': None,
        },
    )


async def get_pod_description(session: ClientSession) -> CallToolResult:
    """Calls the 'get_pod_description' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'get_pod_description' tool.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling '  get_pod_description' tool'")
    return await session.call_tool(
        name='get_pod_description',
        arguments={
            'namespace': 'default',
            'podname': 'test',
            'container': None,
        },
    )


async def get_pod_events(session: ClientSession) -> CallToolResult:
    """Calls the 'get_pod_events' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'get_pod_events' tool.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'get_pod_events' tool'")
    return await session.call_tool(
        name='get_pod_events',
        arguments={
            'namespace': 'default',
            'podname': 'test',
            'container': None,
        },
    )


# Node-related client functions
async def get_node_info(session: ClientSession, node_name: str) -> CallToolResult:
    """Calls the 'get_node_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        node_name: The name of the node.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_node_info' tool for node: {node_name}")
    return await session.call_tool(
        name='get_node_info',
        arguments={
            'node_name': node_name,
        },
    )


async def list_nodes(session: ClientSession) -> CallToolResult:
    """Calls the 'list_nodes' tool on the connected MCP server.

    Args:
        session: The active ClientSession.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'list_nodes' tool")
    return await session.call_tool(
        name='list_nodes',
        arguments={},
    )


async def get_node_metrics(session: ClientSession, node_name: str) -> CallToolResult:
    """Calls the 'get_node_metrics' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        node_name: The name of the node.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_node_metrics' tool for node: {node_name}")
    return await session.call_tool(
        name='get_node_metrics',
        arguments={
            'node_name': node_name,
        },
    )


# Deployment-related client functions
async def get_deployment_info(session: ClientSession, namespace: str, deployment_name: str) -> CallToolResult:
    """Calls the 'get_deployment_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the deployment.
        deployment_name: The name of the deployment.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_deployment_info' tool for deployment: {deployment_name} in namespace: {namespace}")
    return await session.call_tool(
        name='get_deployment_info',
        arguments={
            'namespace': namespace,
            'deployment_name': deployment_name,
        },
    )


async def list_deployments(session: ClientSession, namespace: str) -> CallToolResult:
    """Calls the 'list_deployments' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace to list deployments from.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_deployments' tool for namespace: {namespace}")
    return await session.call_tool(
        name='list_deployments',
        arguments={
            'namespace': namespace,
        },
    )


# Service-related client functions
async def get_service_info(session: ClientSession, namespace: str, service_name: str) -> CallToolResult:
    """Calls the 'get_service_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the service.
        service_name: The name of the service.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_service_info' tool for service: {service_name} in namespace: {namespace}")
    return await session.call_tool(
        name='get_service_info',
        arguments={
            'namespace': namespace,
            'service_name': service_name,
        },
    )


async def list_services(session: ClientSession, namespace: str) -> CallToolResult:
    """Calls the 'list_services' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace to list services from.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_services' tool for namespace: {namespace}")
    return await session.call_tool(
        name='list_services',
        arguments={
            'namespace': namespace,
        },
    )


# Ingress-related client functions
async def get_ingress_info(session: ClientSession, namespace: str, ingress_name: str) -> CallToolResult:
    """Calls the 'get_ingress_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the ingress.
        ingress_name: The name of the ingress.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_ingress_info' tool for ingress: {ingress_name} in namespace: {namespace}")
    return await session.call_tool(
        name='get_ingress_info',
        arguments={
            'namespace': namespace,
            'ingress_name': ingress_name,
        },
    )


async def list_ingresses(session: ClientSession, namespace: str) -> CallToolResult:
    """Calls the 'list_ingresses' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace to list ingresses from.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_ingresses' tool for namespace: {namespace}")
    return await session.call_tool(
        name='list_ingresses',
        arguments={
            'namespace': namespace,
        },
    )


# ConfigMap-related client functions
async def get_configmap_info(session: ClientSession, namespace: str, configmap_name: str) -> CallToolResult:
    """Calls the 'get_configmap_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the configmap.
        configmap_name: The name of the configmap.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_configmap_info' tool for configmap: {configmap_name} in namespace: {namespace}")
    return await session.call_tool(
        name='get_configmap_info',
        arguments={
            'namespace': namespace,
            'configmap_name': configmap_name,
        },
    )


async def list_configmaps(session: ClientSession, namespace: str) -> CallToolResult:
    """Calls the 'list_configmaps' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace to list configmaps from.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_configmaps' tool for namespace: {namespace}")
    return await session.call_tool(
        name='list_configmaps',
        arguments={
            'namespace': namespace,
        },
    )


# Secret-related client functions
async def get_secret_info(session: ClientSession, namespace: str, secret_name: str) -> CallToolResult:
    """Calls the 'get_secret_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the secret.
        secret_name: The name of the secret.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_secret_info' tool for secret: {secret_name} in namespace: {namespace}")
    return await session.call_tool(
        name='get_secret_info',
        arguments={
            'namespace': namespace,
            'secret_name': secret_name,
        },
    )


async def list_secrets(session: ClientSession, namespace: str) -> CallToolResult:
    """Calls the 'list_secrets' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace to list secrets from.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_secrets' tool for namespace: {namespace}")
    return await session.call_tool(
        name='list_secrets',
        arguments={
            'namespace': namespace,
        },
    )


# PersistentVolumeClaim-related client functions
async def get_persistent_volume_claim_info(session: ClientSession, namespace: str, pvc_name: str) -> CallToolResult:
    """Calls the 'get_persistent_volume_claim_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the PVC.
        pvc_name: The name of the PVC.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_persistent_volume_claim_info' tool for PVC: {pvc_name} in namespace: {namespace}")
    return await session.call_tool(
        name='get_persistent_volume_claim_info',
        arguments={
            'namespace': namespace,
            'pvc_name': pvc_name,
        },
    )


async def list_persistent_volume_claims(session: ClientSession, namespace: str) -> CallToolResult:
    """Calls the 'list_persistent_volume_claims' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace to list PVCs from.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_persistent_volume_claims' tool for namespace: {namespace}")
    return await session.call_tool(
        name='list_persistent_volume_claims',
        arguments={
            'namespace': namespace,
        },
    )


# PersistentVolume-related client functions
async def get_persistent_volume_info(session: ClientSession, pv_name: str) -> CallToolResult:
    """Calls the 'get_persistent_volume_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        pv_name: The name of the persistent volume.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_persistent_volume_info' tool for PV: {pv_name}")
    return await session.call_tool(
        name='get_persistent_volume_info',
        arguments={
            'pv_name': pv_name,
        },
    )


async def list_persistent_volumes(session: ClientSession) -> CallToolResult:
    """Calls the 'list_persistent_volumes' tool on the connected MCP server.

    Args:
        session: The active ClientSession.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'list_persistent_volumes' tool")
    return await session.call_tool(
        name='list_persistent_volumes',
        arguments={},
    )


# Namespace-related client functions
async def get_namespace_info(session: ClientSession, namespace_name: str) -> CallToolResult:
    """Calls the 'get_namespace_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace_name: The name of the namespace.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_namespace_info' tool for namespace: {namespace_name}")
    return await session.call_tool(
        name='get_namespace_info',
        arguments={
            'namespace_name': namespace_name,
        },
    )


async def list_namespaces(session: ClientSession) -> CallToolResult:
    """Calls the 'list_namespaces' tool on the connected MCP server.

    Args:
        session: The active ClientSession.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'list_namespaces' tool")
    return await session.call_tool(
        name='list_namespaces',
        arguments={},
    )


# Cluster-wide client functions
async def list_events(session: ClientSession, namespace: str = None, field_selector: str = None) -> CallToolResult:
    """Calls the 'list_events' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: Optional namespace to filter events.
        field_selector: Optional field selector for filtering events.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'list_events' tool with namespace: {namespace}, field_selector: {field_selector}")
    arguments = {}
    if namespace:
        arguments['namespace'] = namespace
    if field_selector:
        arguments['field_selector'] = field_selector
    
    return await session.call_tool(
        name='list_events',
        arguments=arguments,
    )


async def get_cluster_info(session: ClientSession) -> CallToolResult:
    """Calls the 'get_cluster_info' tool on the connected MCP server.

    Args:
        session: The active ClientSession.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'get_cluster_info' tool")
    return await session.call_tool(
        name='get_cluster_info',
        arguments={},
    )


async def get_cluster_component_status(session: ClientSession) -> CallToolResult:
    """Calls the 'get_cluster_component_status' tool on the connected MCP server.

    Args:
        session: The active ClientSession.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'get_cluster_component_status' tool")
    return await session.call_tool(
        name='get_cluster_component_status',
        arguments={},
    )


async def get_api_resources(session: ClientSession) -> CallToolResult:
    """Calls the 'get_api_resources' tool on the connected MCP server.

    Args:
        session: The active ClientSession.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'get_api_resources' tool")
    return await session.call_tool(
        name='get_api_resources',
        arguments={},
    )


async def get_network_policies(session: ClientSession, namespace: str = None) -> CallToolResult:
    """Calls the 'get_network_policies' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: Optional namespace to filter network policies.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_network_policies' tool with namespace: {namespace}")
    arguments = {}
    if namespace:
        arguments['namespace'] = namespace
    
    return await session.call_tool(
        name='get_network_policies',
        arguments=arguments,
    )


# Enhanced pod metrics client function
async def get_pod_metrics(session: ClientSession, namespace: str, pod_name: str, container_name: str = None) -> CallToolResult:
    """Calls the 'get_pod_metrics' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        namespace: The namespace of the pod.
        pod_name: The name of the pod.
        container_name: Optional name of the container in the pod.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'get_pod_metrics' tool for pod: {pod_name} in namespace: {namespace}")
    arguments = {
        'namespace': namespace,
        'pod_name': pod_name,
    }
    if container_name:
        arguments['container_name'] = container_name
    
    return await session.call_tool(
        name='get_pod_metrics',
        arguments=arguments,
    )


# Test util
async def main(host, port, transport, query, resource, tool):
    """Main asynchronous function to connect to the MCP server and execute commands.

    Used for local testing.

    Args:
        host: Server hostname.
        port: Server port.
        transport: Connection transport ('sse' or 'stdio').
        query: Optional query string for the 'find_agent' tool.
        resource: Optional resource URI to read.
    """
    logger.info('Starting Client to connect to MCP')
    async with init_session(host, port, transport) as session:
        if query:
            result = await find_agent(session, query)
            data = json.loads(result.content[0].text)
            logger.info(json.dumps(data, indent=2))
        if resource:
            result = await find_resource(session, resource)
            logger.info(result)
            data = json.loads(result.contents[0].text)
            logger.info(json.dumps(data, indent=2))
        if tool:
            if tool == 'list_pods':
                results = await list_pods(session)
                logger.info(results.model_dump())
            if tool == 'get_pod_description':
                result = await get_pod_description(session)
                data = json.loads(result.content[0].text)
                logger.info(json.dumps(data, indent=2))
            if tool == 'get_pod_events':
                result = await get_pod_events(session)
                logger.info(result)
                data = json.loads(result.content[0].text)
                logger.info(json.dumps(data, indent=2))


# Command line tester
@click.command()
@click.option('--host', default='localhost', help='SSE Host')
@click.option('--port', default='10100', help='SSE Port')
@click.option('--transport', default='stdio', help='MCP Transport')
@click.option('--find_agent', help='Query to find an agent')
@click.option('--resource', help='URI of the resource to locate')
def cli(host, port, transport, find_agent, resource, tool_name):
    """A command-line client to interact with the Agent Cards MCP server."""
    asyncio.run(main(host, port, transport, find_agent, resource, tool_name))


if __name__ == '__main__':
    cli()