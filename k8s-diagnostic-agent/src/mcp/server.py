from mcp.server.fastmcp import FastMCP
from kubernetes import client, config
import time

# Initialize the MCP server
mcp = FastMCP("K8sDiagnostic")

# Load Kubernetes configuration (assumes ~/.kube/config is set up)
try:
    config.load_kube_config()
except Exception as e:
    raise Exception(f"Failed to load Kubernetes config: {str(e)}")

# Create Kubernetes API client
v1 = client.CoreV1Api()

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
        # Check if tcpdump is available in the selected container
        check_cmd = ['/bin/sh', '-c', 'command -v tcpdump || echo "not found"']
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
        
        if "not found" in resp:
            return f"Error: tcpdump is not available in container '{container_name}'. Please ensure the diagnostic tools are installed."
        
        # Build the tcpdump command
        output_file = f"/tmp/capture-{pod_name}-{int(time.time())}.pcap"
        filter_arg = f" '{filter_expr}'" if filter_expr else ""
        
        tcpdump_cmd = f"tcpdump -i {interface} -w {output_file} -G {duration} -W 1{filter_arg}"
        exec_command = ['/bin/sh', '-c', f"nohup {tcpdump_cmd} > /tmp/tcpdump.log 2>&1 &"]
        
        # Execute the command in the container
        v1.connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace=namespace,
            container=selected_container,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )
        
        return f"Tcpdump started in pod '{pod_name}', container '{selected_container}'. Capturing traffic on interface '{interface}' for {duration} seconds. Output will be saved to {output_file}."
    
    except Exception as e:
        return f"Error running tcpdump: {str(e)}"

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
    # Mount path is set to '/diagnostic' for explicit path matching
    print("Starting K8s Diagnostic MCP server on port 8000")
    mcp.run(transport='streamable-http', mount_path='/diagnostic')