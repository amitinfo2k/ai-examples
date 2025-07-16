"""
Kubernetes MCP Server implementation.

This server exposes tools for kubectl commands to interact with a Kubernetes cluster.
"""

import subprocess
import json
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)

# Create an MCP server for Kubernetes
mcp = FastMCP("K8sPodLister")


@mcp.tool()
def get_pod_description(namespace: str, pod_name: str) -> Dict[str, Any]:
    """
    Get the description of a Kubernetes pod in YAML format.
    
    Args:
        namespace: The namespace of the pod
        pod_name: The name of the pod
        
    Returns:
        The pod description in YAML format
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "yaml"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"yaml": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}


@mcp.tool()
def get_pod_events(namespace: str, pod_name: str) -> Dict[str, Any]:
    """
    Get events related to a Kubernetes pod.
    
    Args:
        namespace: The namespace of the pod
        pod_name: The name of the pod
        
    Returns:
        The pod events
    """
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "events",
                "-n",
                namespace,
                "--field-selector",
                f"involvedObject.name={pod_name}",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"events": json.loads(result.stdout)}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON output from kubectl"}


@mcp.tool()
def get_pod_logs(
    namespace: str, pod_name: str, container_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get logs from a Kubernetes pod.
    
    Args:
        namespace: The namespace of the pod
        pod_name: The name of the pod
        container_name: Optional name of the container in the pod
        
    Returns:
        The pod logs
    """
    cmd = ["kubectl", "logs", pod_name, "-n", namespace]
    if container_name:
        cmd.extend(["-c", container_name])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return {"logs": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}


@mcp.tool()
def list_pods(namespace: str) -> Dict[str, Any]:
    """
    List all pods in a namespace.
    
    Args:
        namespace: The namespace to list pods from
        
    Returns:
        List of pods in the namespace
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"pods": json.loads(result.stdout)}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON output from kubectl"}


@mcp.tool()
def describe_node(node_name: str) -> Dict[str, Any]:
    """
    Get the description of a Kubernetes node.
    
    Args:
        node_name: The name of the node
        
    Returns:
        The node description
    """
    try:
        result = subprocess.run(
            ["kubectl", "describe", "node", node_name],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"description": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}


@mcp.tool()
def get_resource_usage(namespace: str, pod_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get resource usage information for pods in a namespace.
    
    Args:
        namespace: The namespace to get resource usage from
        pod_name: Optional name of a specific pod
        
    Returns:
        Resource usage information
    """
    cmd = ["kubectl", "top", "pods", "-n", namespace]
    if pod_name:
        cmd.append(pod_name)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return {"usage": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}


if __name__ == "__main__":
    # Run the server with streamable-http transport
    # Default port is 8000
    # Mount path is set to '/mcp' for explicit path matching
    mcp.run(transport='streamable-http', mount_path='/mcp')
