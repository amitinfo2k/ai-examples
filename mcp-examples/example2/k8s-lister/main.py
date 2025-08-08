from mcp.server.fastmcp import FastMCP
from kubernetes import client, config
import json
import time

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
            print(f"[DEBUG] Service name: {service_name}")
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
            print(f"[DEBUG] Service data: {service_data}")
        
        if service_data:
            print(f"[DEBUG] Service data: {service_data}")
            return json.dumps({"services": service_data}, indent=2)
        else:
            print(f"[DEBUG] No services found in namespace '{namespace}'")
            return json.dumps({"message": f"No services found in namespace '{namespace}'"})
           
        
    except Exception as e:
        return json.dumps({"error": f"Error fetching services: {str(e)}"})      

@mcp.tool()
def run_tcpdump(pod_name: str, container_name: str = None, namespace: str = "default", interface: str = "eth0", duration: int = 30, filter_expr: str = "") -> str:
    """     
    Run tcpdump in the specified container to capture network traffic
    Args:
        pod_name: The name of the pod
        container_name: The name of the container
        namespace: The namespace of the pod
        interface: Network interface to capture traffic on (default: eth0)
        duration: Duration in seconds to capture traffic (default: 30)
        filter_expr: Optional tcpdump filter expression
    """
    print(f"[DEBUG] run_tcpdump called with: pod_name={pod_name}, container_name={container_name}, namespace={namespace}, interface={interface}, duration={duration}")
    try:
        # Verify the pod exists
        print(f"[DEBUG] Verifying pod '{pod_name}' exists in namespace '{namespace}'")
        try:
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            print(f"[DEBUG] Pod found: {pod.metadata.name}")
        except client.exceptions.ApiException as e:
            print(f"[DEBUG] Pod verification failed: {e}")
            if e.status == 404:
                return f"Error: Pod '{pod_name}' not found in namespace '{namespace}'."
            else:
                return f"Error accessing pod: {str(e)}"
        
        # Find the container to use
        selected_container = None
        if container_name:
            print(f"[DEBUG] Looking for container '{container_name}' in pod")
            for container in pod.spec.containers:
                print(f"[DEBUG] Found container: {container.name}")
                if container.name == container_name:
                    selected_container = container.name
                    break
            if not selected_container:
                return f"Error: Container '{container_name}' not found in pod '{pod_name}'."
        else:
            selected_container = pod.spec.containers[0].name
            print(f"[DEBUG] Using first container: {selected_container}")
        
        print(f"[DEBUG] Selected container: {selected_container}")
        # For now, let's skip the tcpdump check and proceed with execution
        print(f"[DEBUG] Skipping tcpdump availability check for now")
        
        # Build the tcpdump command
        output_file = f"/tmp/capture-{pod_name}-{int(time.time())}.pcap"
        filter_arg = f" '{filter_expr}'" if filter_expr else ""
        
        # Skip the test and go directly to kubectl exec approach since connect_get_namespaced_pod_exec has WebSocket issues
        try:
            print(f"[DEBUG] Using kubectl exec approach directly")
            import subprocess
            
            tcpdump_cmd = f"timeout -s INT {duration} tcpdump -i {interface} -w {output_file} {filter_arg}"
            
            kubectl_cmd = [
                'kubectl', 'exec', '-n', namespace, pod_name, 
                '-c', selected_container, '--', '/bin/sh', '-c', 
                f"{tcpdump_cmd} > /tmp/tcpdump.log 2>&1 & echo $!"
            ]
            
            print(f"[DEBUG] Executing kubectl command: {' '.join(kubectl_cmd)}")
            
            result = subprocess.run(kubectl_cmd, capture_output=True, text=True, timeout=30)
            resp = result.stdout
            print(f"[DEBUG] Kubectl exec result: {result.stdout}")
            if result.stderr:
                print(f"[DEBUG] Kubectl exec stderr: {result.stderr}")
                
        except Exception as kubectl_error:
            print(f"[DEBUG] Kubectl exec failed: {kubectl_error}")
            return f"Error: Failed to start tcpdump via kubectl: {str(kubectl_error)}"
            
            print(f"[DEBUG] Tcpdump execution response: {resp}")
            
            # The response should contain the PID of the background process
            if resp and resp.strip().isdigit():
                pid = resp.strip()
                print(f"[DEBUG] Tcpdump started with PID: {pid}")
            else:
                print(f"[DEBUG] Could not determine PID from response: {resp}")
            
            # Verify the process is running
            try:
                verify_cmd = ['/bin/sh', '-c', 'ps aux | grep tcpdump | grep -v grep']
                verify_resp = v1.connect_get_namespaced_pod_exec(
                    name=pod_name,
                    namespace=namespace,
                    container=selected_container,
                    command=verify_cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                print(f"[DEBUG] Process verification: {verify_resp}")
                
                if not verify_resp.strip():
                    return f"Warning: Tcpdump command was executed but no tcpdump process found running. Check /tmp/tcpdump.log for errors."
                
            except Exception as verify_e:
                print(f"[DEBUG] Process verification failed: {verify_e}")
            
        except Exception as e:
            print(f"[DEBUG] Tcpdump execution failed: {e}")
            return f"Error: Failed to start tcpdump: {str(e)}"
        
        return f"Tcpdump started successfully in pod '{pod_name}', container '{selected_container}'. Capturing traffic on interface '{interface}' for {duration} seconds. Output will be saved to {output_file}. Check /tmp/tcpdump.log for execution details."
    
    except Exception as e:
        return f"Error running tcpdump: {str(e)}"

@mcp.tool()
def check_tcpdump_status(pod_name: str, container_name: str = None, namespace: str = "default") -> str:
    """
    Check the status of tcpdump processes and logs in the specified container
    Args:
        pod_name: The name of the pod
        container_name: The name of the container
        namespace: The namespace of the pod
    """
    try:
        # Verify the pod exists
        try:
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                return f"Error: Pod '{pod_name}' not found in namespace '{namespace}'."
            else:
                return f"Error accessing pod: {str(e)}"
        
        # Find the container to use
        selected_container = None
        if container_name:
            for container in pod.spec.containers:
                if container.name == container_name:
                    selected_container = container.name
                    break
            if not selected_container:
                return f"Error: Container '{container_name}' not found in pod '{pod_name}'."
        else:
            selected_container = pod.spec.containers[0].name
        
        results = []
        
        # Check for running tcpdump processes
        try:
            import subprocess
            ps_cmd = ['kubectl', 'exec', '-n', namespace, pod_name, '-c', selected_container, '--', '/bin/sh', '-c', 'ps aux | grep tcpdump | grep -v grep']
            result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=30)
            ps_resp = result.stdout
            if result.stderr:
                ps_resp += f"\nStderr: {result.stderr}"
            results.append(f"Running tcpdump processes:\n{ps_resp}\n")
        except Exception as e:
            results.append(f"Could not check running processes: {e}\n")
        
        # Check tcpdump log file
        try:
            import subprocess
            log_cmd = ['kubectl', 'exec', '-n', namespace, pod_name, '-c', selected_container, '--', '/bin/sh', '-c', 'cat /tmp/tcpdump.log 2>/dev/null || echo "Log file not found"']
            result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=30)
            log_resp = result.stdout
            if result.stderr:
                log_resp += f"\nStderr: {result.stderr}"
            results.append(f"Tcpdump log file contents:\n{log_resp}\n")
        except Exception as e:
            results.append(f"Could not read log file: {e}\n")
        
        # Check for pcap files
        try:
            import subprocess
            pcap_cmd = ['kubectl', 'exec', '-n', namespace, pod_name, '-c', selected_container, '--', '/bin/sh', '-c', 'ls -la /tmp/capture-*.pcap 2>/dev/null || echo "No pcap files found"']
            result = subprocess.run(pcap_cmd, capture_output=True, text=True, timeout=30)
            pcap_resp = result.stdout
            if result.stderr:
                pcap_resp += f"\nStderr: {result.stderr}"
            results.append(f"Pcap files in /tmp:\n{pcap_resp}\n")
        except Exception as e:
            results.append(f"Could not check pcap files: {e}\n")
        
        return "".join(results)
        
    except Exception as e:
        return f"Error checking tcpdump status: {str(e)}"

@mcp.tool()
def run_dns_tool(pod_name: str, container_name: str = None, namespace: str = "default", target: str = "kubernetes.default.svc.cluster.local", query_type: str = "A") -> str:
    """     
    Run DNS lookup tools in the specified container
    Args:
        pod_name: The name of the pod
        container_name: The name of the container
        namespace: The namespace of the pod
        target: DNS name to query (default: kubernetes.default.svc.cluster.local)
        query_type: DNS query type (A, AAAA, MX, etc.) (default: A)
    """
    try:
        # Verify the pod exists
        try:
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                return f"Error: Pod '{pod_name}' not found in namespace '{namespace}'."
            else:
                return f"Error accessing pod: {str(e)}"
        
        # Find the container to use
        selected_container = None
        if container_name:
            for container in pod.spec.containers:
                if container.name == container_name:
                    selected_container = container.name
                    break
            if not selected_container:
                return f"Error: Container '{container_name}' not found in pod '{pod_name}'."
        else:
            # Fallback to first container in the pod
            selected_container = pod.spec.containers[0].name
        
        # Check if DNS tools are available
        dns_tools = {
            "dig": False,
            "nslookup": False
        }
        
        for tool in dns_tools:
            check_cmd = ['/bin/sh', '-c', f'command -v {tool} || echo "not found"']
            resp = v1.connect_get_namespaced_pod_exec(
                name=pod_name,
                namespace=namespace,
                container=selected_container,
                command=check_cmd,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            
            if "not found" not in resp:
                dns_tools[tool] = True
        
        if not any(dns_tools.values()):
            return f"Error: No DNS tools (dig, nslookup) are available in container '{selected_container}'. Please ensure the diagnostic tools are installed."
        
        # Run available DNS lookup commands
        results = []
        
        # Add resolv.conf check
        try:
            resolv_cmd = ['/bin/sh', '-c', 'cat /etc/resolv.conf']
            resp = v1.connect_get_namespaced_pod_exec(
                name=pod_name,
                namespace=namespace,
                container=selected_container,
                command=resolv_cmd,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            results.append(f"DNS Configuration (/etc/resolv.conf):\n{resp}\n")
        except Exception:
            results.append("Could not read DNS configuration (/etc/resolv.conf)\n")
        
        # Run dig if available
        if dns_tools["dig"]:
            dig_cmd = ['/bin/sh', '-c', f'dig {target} {query_type}']
            try:
                resp = v1.connect_get_namespaced_pod_exec(
                    name=pod_name,
                    namespace=namespace,
                    container=selected_container,
                    command=dig_cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                results.append(f"Dig Results:\n{resp}\n")
            except Exception as e:
                results.append(f"Error running dig: {str(e)}\n")
        
        # Run nslookup if available
        if dns_tools["nslookup"]:
            nslookup_cmd = ['/bin/sh', '-c', f'nslookup -type={query_type} {target}']
            try:
                resp = v1.connect_get_namespaced_pod_exec(
                    name=pod_name,
                    namespace=namespace,
                    container=selected_container,
                    command=nslookup_cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                results.append(f"Nslookup Results:\n{resp}\n")
            except Exception as e:
                results.append(f"Error running nslookup: {str(e)}\n")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error running DNS tools: {str(e)}"

@mcp.tool()
def collect_diagnostic_data(pod_name: str, container_name: str = None, namespace: str = "default") -> str:
    """     
    Collect comprehensive diagnostic data from the specified container
    Args:
        pod_name: The name of the pod
        container_name: The name of the container
        namespace: The namespace of the pod
    """
    try:
        # Verify the pod exists
        try:
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                return f"Error: Pod '{pod_name}' not found in namespace '{namespace}'."
            else:
                return f"Error accessing pod: {str(e)}"
        
        # Find the container to use
        selected_container = None
        if container_name:
            for container in pod.spec.containers:
                if container.name == container_name:
                    selected_container = container.name
                    break
            if not selected_container:
                return f"Error: Container '{container_name}' not found in pod '{pod_name}'."
        else:
            selected_container = pod.spec.containers[0].name
        # Create a directory for diagnostic data
        timestamp = int(time.time())
        diagnostic_dir = f"/tmp/diagnostic-{pod_name}-{timestamp}"
        mkdir_cmd = ['/bin/sh', '-c', f"mkdir -p {diagnostic_dir}"]
        v1.connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace=namespace,
            container=selected_container,
            command=mkdir_cmd,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )
        
        # Collect diagnostic data
        diagnostic_files = []
        
        # Define diagnostic commands to run
        diagnostic_commands = {
            # Network diagnostics
            "netstat": f"netstat -tunapl > {diagnostic_dir}/netstat.txt 2>/dev/null || echo 'netstat not available' > {diagnostic_dir}/netstat.txt",
            "ip_addr": f"ip addr > {diagnostic_dir}/ip-addr.txt 2>/dev/null || echo 'ip addr not available' > {diagnostic_dir}/ip-addr.txt",
            "ip_route": f"ip route > {diagnostic_dir}/ip-route.txt 2>/dev/null || echo 'ip route not available' > {diagnostic_dir}/ip-route.txt",
            "dns_config": f"cat /etc/resolv.conf > {diagnostic_dir}/resolv.conf.txt 2>/dev/null || echo 'resolv.conf not available' > {diagnostic_dir}/resolv.conf.txt",
            
            # System diagnostics
            "processes": f"ps aux > {diagnostic_dir}/processes.txt 2>/dev/null || echo 'ps not available' > {diagnostic_dir}/processes.txt",
            "memory": f"free -m > {diagnostic_dir}/memory.txt 2>/dev/null || echo 'free not available' > {diagnostic_dir}/memory.txt",
            "disk": f"df -h > {diagnostic_dir}/disk-usage.txt 2>/dev/null || echo 'df not available' > {diagnostic_dir}/disk-usage.txt",
            "env_vars": f"env > {diagnostic_dir}/environment.txt 2>/dev/null || echo 'env not available' > {diagnostic_dir}/environment.txt"
        }
        
        # Execute each diagnostic command
        for name, cmd in diagnostic_commands.items():
            try:
                exec_cmd = ['/bin/sh', '-c', cmd]
                v1.connect_get_namespaced_pod_exec(
                    name=pod_name,
                    namespace=namespace,
                    container=selected_container,
                    command=exec_cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                diagnostic_files.append(name)
            except Exception:
                # Continue with other commands if one fails
                pass
        
        # Get container logs
        try:
            logs = v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=selected_container,
                tail_lines=1000
            )
            
            log_file = f"{diagnostic_dir}/container-logs.txt"
            write_cmd = ['/bin/sh', '-c', f"cat > {log_file} << 'EOL'\n{logs}\nEOL"]
            v1.connect_get_namespaced_pod_exec(
                name=pod_name,
                namespace=namespace,
                container=selected_container,
                command=write_cmd,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            diagnostic_files.append("container-logs")
        except Exception:
            # Continue if logs can't be collected
            pass
        
        # Create a summary file
        summary = f"Diagnostic data collected at {timestamp}\n"
        summary += f"Pod: {pod_name}\n"
        summary += f"Container: {container_name}\n"
        summary += f"Namespace: {namespace}\n"
        summary += "Files collected:\n"
        for file in diagnostic_files:
            summary += f"- {file}\n"
        
        summary_file = f"{diagnostic_dir}/summary.txt"
        write_summary_cmd = ['/bin/sh', '-c', f"cat > {summary_file} << 'EOL'\n{summary}\nEOL"]
        v1.connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace=namespace,
            container=selected_container,
            command=write_summary_cmd,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )
        
        return f"Diagnostic data collected and saved to {diagnostic_dir}\n{summary}"
    
    except Exception as e:
        return f"Error collecting diagnostic data: {str(e)}"  

if __name__ == "__main__":
    # Run the server with streamable-http transport
    # Default port is 8000
    # Mount path is set to '/mcp' for explicit path matching
    mcp.run(transport='streamable-http', mount_path='/mcp')
    #list_services(namespace="default")