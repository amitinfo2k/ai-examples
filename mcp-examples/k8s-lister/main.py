from mcp.server.fastmcp import FastMCP
from kubernetes import client, config
import json

# Initialize the MCP server
mcp = FastMCP("K8sPodLister")

# Load Kubernetes configuration (assumes ~/.kube/config is set up)
try:
    config.load_kube_config()
except Exception as e:
    raise Exception(f"Failed to load Kubernetes config: {str(e)}")

# Create Kubernetes API client
v1 = client.CoreV1Api()

@mcp.tool()
def list_pods(namespace: str = "default") -> str:
    """
    List all pods in the specified Kubernetes namespace and their resource usage in a tabular format.
    Args:
        namespace: The Kubernetes namespace to query (default: 'default')
    Returns:
        A string containing a table of pods with their name, status, CPU, and memory requests.
    """
    try:
        # Fetch pods from the specified namespace
        pods = v1.list_namespaced_pod(namespace=namespace).items
        
        # Prepare table data
        pod_data = []
        for pod in pods:
            pod_name = pod.metadata.name
            status = pod.status.phase
            cpu_request = "N/A"
            memory_request = "N/A"
            
            for container in pod.spec.containers:
                if container.resources and container.resources.requests:
                    cpu_request = container.resources.requests.get("cpu", "N/A")
                    memory_request = container.resources.requests.get("memory", "N/A")
            pod_data.append({
                "pod_name": pod_name,
                "status": status,
                "cpu_request": cpu_request,
                "memory_request": memory_request
            })
        if pod_data:
            return json.dumps({"pods": pod_data}, indent=2)
        else:
            return json.dumps({"message": f"No pods found in namespace '{namespace}'"})
    except Exception as e:
        return json.dumps({"error": f"Error: fetching pods: {str(e)}"})


@mcp.tool()
def list_services(namespace: str = "default") -> str:
    """
    List all services in the specified Kubernetes namespace and their resource usage in a tabular format.
    Args:
        namespace: The Kubernetes namespace to query (default: 'default')
    Returns:
        A string containing a table of services with their name, status, cluster IP, ports, node port, and selector.
    """
    try:
        # Fetch services from the specified namespace
        services = v1.list_namespaced_service(namespace=namespace).items
        print(f"[DEBUG] Found {len(services)} services in namespace '{namespace}'")
        
        # Prepare table data
        service_data = []
        for service in services:
            service_name = service.metadata.name
            # Services don't have a status.phase field like pods do
            status = "Active" if service.metadata.deletion_timestamp is None else "Terminating"
            cluster_ip = "N/A"
            ports_list = []
            node_ports = []
            selector = "N/A"
            
            if service.spec.cluster_ip:
                cluster_ip = service.spec.cluster_ip
            
            if service.spec.ports:
                for port in service.spec.ports:
                    ports_list.append(f"{port.port}/{port.protocol}")
                    if port.node_port:
                        node_ports.append(str(port.node_port))
                
                ports = ", ".join(ports_list) if ports_list else "N/A"
                node_port = ", ".join(node_ports) if node_ports else "N/A"
            
            if service.spec.selector:
                selector_items = [f"{k}={v}" for k, v in service.spec.selector.items()]
                selector = ", ".join(selector_items)
            
            service_data.append({
                "service_name": service_name,
                "status": status,
                "cluster_ip": cluster_ip,
                "ports": ports,
                "node_port": node_port,
                "selector": selector
            })
        
        if service_data:
            return json.dumps({"services": service_data}, indent=2)
        else:
            return json.dumps({"message": f"No services found in namespace '{namespace}'"})
            
    except Exception as e:
        return json.dumps({"error": f"Error fetching services: {str(e)}"})        

if __name__ == "__main__":
    # Run the server with stdio transport
    mcp.run(transport='stdio')