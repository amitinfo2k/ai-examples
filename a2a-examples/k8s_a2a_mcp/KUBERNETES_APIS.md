# Kubernetes APIs for MCP Server

This document provides a comprehensive guide to the Kubernetes APIs available in the MCP server for debugging and monitoring Kubernetes clusters.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [API Categories](#api-categories)
5. [Usage Examples](#usage-examples)
6. [Testing](#testing)
7. [Error Handling](#error-handling)
8. [Security & Performance](#security--performance)
9. [Troubleshooting](#troubleshooting)
10. [Integration with AI Agents](#integration-with-ai-agents)

## Overview

The MCP server provides a wide range of Kubernetes APIs that allow AI agents to:
- Gather comprehensive information about cluster resources
- Monitor real-time metrics and logs
- Diagnose issues across different layers (application, pod, node, network, storage, cluster)
- Correlate events and metrics for better problem identification

## Prerequisites

1. **Kubernetes Cluster**: A running Kubernetes cluster with kubectl configured
2. **Python Environment**: Python 3.8+ with required dependencies
3. **MCP Server**: The MCP server must be running and accessible
4. **RBAC Permissions**: Appropriate Kubernetes RBAC permissions for the MCP server

## Installation & Setup

1. Navigate to the project directory:
   ```bash
   cd a2a-examples/k8s_a2a_mcp
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up your environment:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   ```

4. Start the MCP server:
   ```bash
   uv run python -m k8s_debug.mcp.server
   ```

## API Categories

### 1. Node-related APIs

#### `get_node_info(node_name: str)`
Get detailed information about a Kubernetes node including capacity, usage, and status.

**Use case**: Diagnose node-level issues like resource exhaustion, network problems, or disk issues.

#### `list_nodes()`
List all nodes in the cluster with their status (Ready, NotReady).

**Use case**: Get an overview of cluster infrastructure and identify widespread issues.

#### `get_node_metrics(node_name: str)`
Get resource metrics for a specific node (CPU, memory, disk usage).

**Use case**: Monitor node resource pressure and identify nodes under stress.

### 2. Pod-related APIs

#### `list_pods(namespace: str)`
List all pods in a namespace.

#### `get_pod_description(namespace: str, pod_name: str)`
Get detailed pod information in YAML format.

#### `get_pod_events(namespace: str, pod_name: str)`
Get events related to a specific pod.

#### `get_pod_logs(namespace: str, pod_name: str, container_name: Optional[str] = None)`
Get logs from a pod or specific container.

**Use case**: Primary source for application-level error diagnosis.

#### `get_pod_metrics(namespace: str, pod_name: str, container_name: Optional[str] = None)`
Get resource metrics for a specific pod and optionally a specific container.

**Use case**: Monitor resource consumption and identify performance issues.

### 3. Deployment-related APIs

#### `get_deployment_info(namespace: str, deployment_name: str)`
Get detailed information about a Kubernetes deployment.

#### `list_deployments(namespace: str)`
List all deployments in a namespace.

**Use case**: Understand deployment state, replica count, update strategy, and rollout history.

### 4. Service-related APIs

#### `get_service_info(namespace: str, service_name: str)`
Get detailed information about a Kubernetes service.

#### `list_services(namespace: str)`
List all services in a namespace.

**Use case**: Diagnose network connectivity issues and service configuration problems.

### 5. Ingress-related APIs

#### `get_ingress_info(namespace: str, ingress_name: str)`
Get detailed information about a Kubernetes ingress.

#### `list_ingresses(namespace: str)`
List all ingresses in a namespace.

**Use case**: Troubleshoot external access issues and ingress rule problems.

### 6. Configuration APIs

#### `get_configmap_info(namespace: str, configmap_name: str)`
Get detailed information about a Kubernetes configmap.

#### `list_configmaps(namespace: str)`
List all configmaps in a namespace.

**Use case**: Diagnose configuration-related application failures.

#### `get_secret_info(namespace: str, secret_name: str)`
Get detailed information about a Kubernetes secret (without revealing actual values).

#### `list_secrets(namespace: str)`
List all secrets in a namespace.

**Use case**: Troubleshoot authentication and credential issues.

### 7. Storage APIs

#### `get_persistent_volume_claim_info(namespace: str, pvc_name: str)`
Get detailed information about a Kubernetes persistent volume claim.

#### `list_persistent_volume_claims(namespace: str)`
List all persistent volume claims in a namespace.

#### `get_persistent_volume_info(pv_name: str)`
Get detailed information about a Kubernetes persistent volume.

#### `list_persistent_volumes()`
List all persistent volumes in the cluster.

**Use case**: Diagnose storage-related issues for stateful applications.

### 8. Namespace APIs

#### `get_namespace_info(namespace_name: str)`
Get detailed information about a Kubernetes namespace.

#### `list_namespaces()`
List all namespaces in the cluster.

**Use case**: Understand resource isolation and quotas.

### 9. Cluster-wide APIs

#### `list_events(namespace: Optional[str] = None, field_selector: Optional[str] = None)`
List events in the cluster with optional filtering.

**Use case**: Look for cluster-wide events indicating broader problems.

#### `get_cluster_info()`
Get high-level information about the Kubernetes cluster.

#### `get_cluster_component_status()`
Get the health status of core Kubernetes components.

**Use case**: Check scheduler, controller-manager, and etcd health.

#### `get_api_resources()`
List all available API resources in the cluster.

**Use case**: Understand cluster capabilities and available resource types.

#### `get_network_policies(namespace: Optional[str] = None)`
List network policies in the cluster or a specific namespace.

**Use case**: Diagnose network policy-related connectivity issues.

### 10. Resource Usage APIs

#### `get_resource_usage(namespace: str, pod_name: Optional[str] = None)`
Get resource usage information for pods in a namespace.

## Usage Examples

### Basic Pod Diagnosis
```python
# Get pod information
pod_info = await get_pod_description(session, "default", "my-pod")

# Get pod logs
pod_logs = await get_pod_logs(session, "default", "my-pod")

# Get pod metrics
pod_metrics = await get_pod_metrics(session, "default", "my-pod")

# Get pod events
pod_events = await get_pod_events(session, "default", "my-pod")
```

### Node-level Investigation
```python
# List all nodes
nodes = await list_nodes(session)

# Get specific node info
node_info = await get_node_info(session, "worker-node-1")

# Get node metrics
node_metrics = await get_node_metrics(session, "worker-node-1")
```

### Deployment Troubleshooting
```python
# Get deployment info
deployment_info = await get_deployment_info(session, "default", "my-deployment")

# List all deployments
deployments = await list_deployments(session, "default")
```

### Network Diagnosis
```python
# Get service info
service_info = await get_service_info(session, "default", "my-service")

# Get ingress info
ingress_info = await get_ingress_info(session, "default", "my-ingress")

# Check network policies
network_policies = await get_network_policies(session, "default")
```

### Storage Investigation
```python
# Get PVC info
pvc_info = await get_persistent_volume_claim_info(session, "default", "my-pvc")

# Get PV info
pv_info = await get_persistent_volume_info(session, "my-pv")
```

### Cluster-wide Monitoring
```python
# Get cluster info
cluster_info = await get_cluster_info(session)

# Get component status
component_status = await get_cluster_component_status(session)

# List events with filtering
events = await list_events(session, namespace="default", field_selector="type=Warning")
```

## Error Handling

All APIs return consistent error responses:
- **Success**: Returns the requested data in a structured format
- **Error**: Returns `{"error": "error_message"}` for kubectl failures or JSON parsing errors

## Security & Performance

### Security Considerations

1. **RBAC Permissions**: The MCP server needs appropriate Kubernetes RBAC permissions to access these APIs
2. **Secret Handling**: Secret APIs return metadata only, not actual secret values
3. **Namespace Isolation**: Most APIs respect namespace boundaries
4. **Read-only Operations**: All APIs are read-only for safety

### Performance Considerations

1. **Rate Limiting**: Respect Kubernetes API server rate limits
2. **Caching**: Consider caching frequently accessed data
3. **Selective Queries**: Use field selectors and namespaces to limit data scope
4. **Async Operations**: All APIs are asynchronous for better performance

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure the MCP server is running and accessible
2. **Permission Errors**: Check Kubernetes RBAC permissions
3. **Import Errors**: Make sure you're running from the correct directory
4. **kubectl Errors**: Verify kubectl is configured and working

### Debug Mode

Enable debug logging by setting the log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with AI Agents

These APIs enable AI agents to:

1. **Comprehensive Context**: Build a holistic view of the cluster
2. **Problem Locating**: Identify issues across different layers
3. **Diagnosis and Correlation**: Correlate events from multiple sources
4. **Proactive Monitoring**: Identify issues before they cause outages
5. **Guided Remediation**: Provide actionable recommendations

## Future Enhancements

Consider adding these advanced APIs in the future:
- `exec_in_pod()`: Execute diagnostic commands in pods (with extreme caution)
- `port_forward()`: Create port forwards for debugging
- `apply_resource()`: Apply resource configurations
- `delete_resource()`: Delete resources (with proper safeguards)
- `scale_deployment()`: Scale deployments up/down
- `rollback_deployment()`: Rollback failed deployments 