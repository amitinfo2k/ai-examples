import os
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# --- Configuration ---
# Replace with the name of the Pod you want to troubleshoot
TARGET_POD_NAME = "crashloop-pod"
# Replace with the namespace where the Pod is running
TARGET_NAMESPACE = "default"
# Image for your diagnostic sidecar (ensure it's accessible in your cluster)
DIAGNOSTIC_SIDECAR_IMAGE = "amitinfo2k/diagnostic-agent:latest"
# Name for the diagnostic sidecar container
DIAGNOSTIC_SIDECAR_NAME = "diagnostic-sidecar"
# Name for the shared volume
DIAGNOSTIC_VOLUME_NAME = "diagnostic-data"
# Mount path for the shared volume inside the sidecar
DIAGNOSTIC_VOLUME_MOUNT_PATH = "/tmp/diagnostic-data"

# --- Kubernetes API Client Initialization ---
def initialize_k8s_client():
    """Initializes the Kubernetes client.
    Attempts to load in-cluster config first, then kubeconfig from default path.
    """
    try:
        config.load_incluster_config()
        print("Using in-cluster Kubernetes configuration.")
    except config.ConfigException:
        try:
            config.load_kube_config()
            print("Using kubeconfig file for Kubernetes configuration.")
        except config.ConfigException:
            raise Exception("Could not configure Kubernetes client. "
                            "Ensure you are running inside a cluster or have a valid kubeconfig.")
    return client.CoreV1Api()

# --- Sidecar Injection Logic ---
def inject_sidecar_to_pod(api_client, pod_name, namespace):
    """
    Fetches an existing Pod's definition, modifies it to include a diagnostic sidecar,
    deletes the original Pod, and creates the new, modified Pod.
    If the pod already has the diagnostic sidecar, it will not be added again.
    """
    print(f"\n--- Attempting to inject sidecar into Pod: {pod_name} in namespace: {namespace} ---")

    try:
        # 1. Get the current Pod definition
        print(f"Fetching current definition for Pod '{pod_name}'...")
        existing_pod = api_client.read_namespaced_pod(name=pod_name, namespace=namespace)
        print("Pod definition fetched successfully.")
        
        # Check if the diagnostic sidecar already exists in the pod
        if existing_pod.spec.containers:
            for container in existing_pod.spec.containers:
                print("Container name:",container.name)
                if container.name == DIAGNOSTIC_SIDECAR_NAME:
                    print(f"Diagnostic sidecar '{DIAGNOSTIC_SIDECAR_NAME}' already exists in pod '{pod_name}'.")
                    print("Skipping sidecar injection.")
                    return existing_pod
        
        print(f"Diagnostic sidecar '{DIAGNOSTIC_SIDECAR_NAME}' not found in pod '{pod_name}'. Proceeding with injection.")


        # Create a new Pod object from the existing one to modify
        new_pod_body = client.V1Pod(
            api_version=existing_pod.api_version,
            kind=existing_pod.kind,
            metadata=client.V1ObjectMeta(
                name=existing_pod.metadata.name,
                namespace=existing_pod.metadata.namespace,
                labels=existing_pod.metadata.labels,
                annotations=existing_pod.metadata.annotations,
                # Clear resourceVersion and uid to avoid conflicts when creating a new pod
                resource_version=None,
                uid=None
            ),
            spec=client.V1PodSpec(
                containers=existing_pod.spec.containers,
                volumes=existing_pod.spec.volumes,
                service_account_name=existing_pod.spec.service_account_name,
                node_name=existing_pod.spec.node_name, # Pin to same node if desired
                affinity=existing_pod.spec.affinity,
                tolerations=existing_pod.spec.tolerations,
                # Crucial for sidecar to share network and process namespace
                share_process_namespace=True,
                # Add other spec fields as needed from the original pod
            )
        )

        # 2. Modify the Pod definition to include the diagnostic sidecar
        print("Adding diagnostic sidecar to the Pod definition...")

        # Define the diagnostic sidecar container
        diagnostic_container = client.V1Container(
            name=DIAGNOSTIC_SIDECAR_NAME,
            image=DIAGNOSTIC_SIDECAR_IMAGE,
            command=["tail", "-f", "/dev/null"], # Keep the sidecar running
            resources=client.V1ResourceRequirements(
                limits={"memory": "64Mi", "cpu": "200m"},
                requests={"memory": "32Mi", "cpu": "100m"}
            ),
            security_context=client.V1SecurityContext(
                privileged=True # WARNING: Grants broad permissions. Use with caution.
                # More secure alternative for tcpdump:
                # capabilities=client.V1Capabilities(add=["NET_RAW", "NET_ADMIN"])
            ),
            volume_mounts=[
                client.V1VolumeMount(
                    name=DIAGNOSTIC_VOLUME_NAME,
                    mount_path=DIAGNOSTIC_VOLUME_MOUNT_PATH
                )
            ]
        )
        new_pod_body.spec.containers.append(diagnostic_container)

        # Define the shared emptyDir volume if it doesn't already exist
        if not new_pod_body.spec.volumes:
            new_pod_body.spec.volumes = []

        volume_exists = False
        for vol in new_pod_body.spec.volumes:
            if vol.name == DIAGNOSTIC_VOLUME_NAME:
                volume_exists = True
                break
        if not volume_exists:
            diagnostic_volume = client.V1Volume(
                name=DIAGNOSTIC_VOLUME_NAME,
                empty_dir=client.V1EmptyDirVolumeSource()
            )
            new_pod_body.spec.volumes.append(diagnostic_volume)

        # 3. Delete the original Pod
        print(f"Deleting original Pod '{pod_name}'... (This will cause brief downtime)")
        api_client.delete_namespaced_pod(name=pod_name, namespace=namespace, body=client.V1DeleteOptions())

        # Wait for the Pod to be deleted
        wait_for_pod_deletion(api_client, pod_name, namespace)
        print(f"Original Pod '{pod_name}' deleted.")

        # 4. Create the new Pod with the sidecar
        print(f"Creating new Pod '{pod_name}' with diagnostic sidecar...")
        created_pod = api_client.create_namespaced_pod(body=new_pod_body, namespace=namespace)
        print(f"New Pod '{created_pod.metadata.name}' created successfully.")

        # Wait for the new Pod to be ready
        wait_for_pod_ready(api_client, created_pod.metadata.name, namespace)
        print(f"New Pod '{created_pod.metadata.name}' is Running and Ready.")
        print("Sidecar injection complete. You can now exec into the 'diagnostic-sidecar' container.")
        
        # Return None to indicate a new pod was created with the sidecar
        return None

    except ApiException as e:
        print(f"Kubernetes API Error during injection: {e}")
        if e.status == 404:
            print(f"Error: Pod '{pod_name}' not found in namespace '{namespace}'.")
        elif e.status == 409: # Conflict, usually means resourceVersion issue if not cleared
            print(f"Error: Conflict when creating Pod. Check if resourceVersion was cleared correctly. {e}")
        else:
            print(f"An unexpected API error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def wait_for_pod_deletion(api_client, pod_name, namespace, timeout_seconds=120):
    """Waits for a Pod to be deleted."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            api_client.read_namespaced_pod(name=pod_name, namespace=namespace)
            time.sleep(2) # Check every 2 seconds
        except ApiException as e:
            if e.status == 404: # Pod not found, means it's deleted
                return
            raise # Re-raise other API exceptions
    raise TimeoutError(f"Timeout waiting for Pod '{pod_name}' to be deleted.")

def wait_for_pod_ready(api_client, pod_name, namespace, timeout_seconds=300):
    """Waits for a Pod to reach 'Running' state and all containers to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            pod = api_client.read_namespaced_pod(name=pod_name, namespace=namespace)
            if pod.status and pod.status.phase == "Running":
                all_containers_ready = True
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if not container_status.ready:
                            all_containers_ready = False
                            break
                if all_containers_ready:
                    return # Pod is running and all containers are ready
            time.sleep(5) # Check every 5 seconds
        except ApiException as e:
            if e.status == 404:
                print(f"Pod '{pod_name}' not found during readiness check. It might have been deleted.")
                return # Or raise an error if this is unexpected
            raise
        except Exception as e:
            print(f"Error during pod readiness check: {e}")
            time.sleep(5)
    raise TimeoutError(f"Timeout waiting for Pod '{pod_name}' to be ready.")


# --- Diagnostic Execution ---
def run_diagnostic_sidecar(api_client, pod_name, namespace):
    """
    Executes diagnostic commands in the sidecar container after it has been injected.
    This function can be customized to run specific diagnostic tools based on your needs.
    
    Args:
        api_client: The Kubernetes API client
        pod_name: Name of the pod containing the diagnostic sidecar
        namespace: Namespace where the pod is running
    """
    print(f"\n--- Running diagnostics in sidecar for Pod: {pod_name} in namespace: {namespace} ---")
    
    try:
        # Check if the sidecar container is ready
        pod = api_client.read_namespaced_pod(name=pod_name, namespace=namespace)
        sidecar_ready = False
        
        if pod.status and pod.status.container_statuses:
            for container_status in pod.status.container_statuses:
                if container_status.name == DIAGNOSTIC_SIDECAR_NAME and container_status.ready:
                    sidecar_ready = True
                    break
        
        if not sidecar_ready:
            print(f"Diagnostic sidecar '{DIAGNOSTIC_SIDECAR_NAME}' is not ready yet. Waiting...")
            wait_for_pod_ready(api_client, pod_name, namespace)
            print(f"Diagnostic sidecar '{DIAGNOSTIC_SIDECAR_NAME}' is now ready.")
        
        print("Diagnostic sidecar is ready for use. You can now execute diagnostic commands.")
        print(f"Example: kubectl exec -it {pod_name} -c {DIAGNOSTIC_SIDECAR_NAME} -n {namespace} -- /bin/sh")
        
        # You can add automatic diagnostic commands here if needed
        # For example, automatically capturing network traffic:
        exec_command = [
            '/bin/sh', 
            '-c', 
            f'tcpdump -i eth0 -w {DIAGNOSTIC_VOLUME_MOUNT_PATH}/capture.pcap -G 30 -W 1'
        ]
        
        print("Running basic diagnostic check...")
        resp = client.CoreV1Api().connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace=namespace,
            container=DIAGNOSTIC_SIDECAR_NAME,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )
        print(f"Diagnostic check result: {resp}")
        
    except ApiException as e:
        print(f"Kubernetes API Error during diagnostic execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during diagnostic execution: {e}")

# --- Download Diagnostic Data ---
def download_diagnostic_data(api_client, pod_name, namespace):
    """
    Downloads diagnostic data collected by the sidecar container to the local machine.
    This function will copy files from the shared volume to the local filesystem.
    
    Args:
        api_client: The Kubernetes API client
        pod_name: Name of the pod containing the diagnostic sidecar
        namespace: Namespace where the pod is running
    """
    print(f"\n--- Downloading diagnostic data from Pod: {pod_name} in namespace: {namespace} ---")
    
    try:
        # Create a local directory to store the diagnostic data
        local_dir = f"./diagnostic-data-{pod_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(local_dir, exist_ok=True)
        print(f"Created local directory: {local_dir}")
        
        # List files in the diagnostic volume
        exec_command = [
            '/bin/sh', 
            '-c', 
            f'ls -la {DIAGNOSTIC_VOLUME_MOUNT_PATH}'
        ]
        
        print("Listing files in the diagnostic volume...")
        file_list_output = client.CoreV1Api().connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace=namespace,
            container=DIAGNOSTIC_SIDECAR_NAME,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )
        print(f"Files in diagnostic volume:\n{file_list_output}")
        
        # Use kubectl cp to download the files
        # Note: We're using os.system here for simplicity, but in a production environment,
        # you might want to use subprocess.run or a similar approach for better error handling
        print("\nDownloading diagnostic files...")
        
        # Check if capture.pcap exists
        check_command = [
            '/bin/sh', 
            '-c', 
            f'[ -f {DIAGNOSTIC_VOLUME_MOUNT_PATH}/capture.pcap ] && echo "exists" || echo "not found"'
        ]
        
        file_exists = client.CoreV1Api().connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace=namespace,
            container=DIAGNOSTIC_SIDECAR_NAME,
            command=check_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        ).strip()
        
        if file_exists == "exists":
            # Use kubectl cp to download the pcap file
            kubectl_cp_cmd = f"kubectl cp {namespace}/{pod_name}:{DIAGNOSTIC_VOLUME_MOUNT_PATH}/capture.pcap {local_dir}/capture.pcap -c {DIAGNOSTIC_SIDECAR_NAME}"
            print(f"Executing: {kubectl_cp_cmd}")
            cp_result = os.system(kubectl_cp_cmd)
            
            if cp_result == 0:
                print(f"Successfully downloaded capture.pcap to {local_dir}/capture.pcap")
            else:
                print(f"Failed to download capture.pcap. Command exit code: {cp_result}")
        else:
            print("No capture.pcap file found in the diagnostic volume.")
            
        # Download any other diagnostic files that might be present
        # This is a simple approach - in a real-world scenario, you might want to
        # parse the file_list_output and download each file individually
        
        print(f"\nDiagnostic data download complete. Files are available in: {local_dir}")
        
    except ApiException as e:
        print(f"Kubernetes API Error during data download: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data download: {e}")

# --- Remove Sidecar ---
def remove_sidecar_from_pod(api_client, pod_name, namespace):
    """
    Removes the diagnostic sidecar from the pod, restoring it to its original state.
    This function will delete the pod with the sidecar and recreate it without the sidecar.
    
    Args:
        api_client: The Kubernetes API client
        pod_name: Name of the pod containing the diagnostic sidecar
        namespace: Namespace where the pod is running
    """
    print(f"\n--- Removing diagnostic sidecar from Pod: {pod_name} in namespace: {namespace} ---")
    
    try:
        # 1. Get the current Pod definition with the sidecar
        print(f"Fetching current definition for Pod '{pod_name}'...")
        existing_pod = api_client.read_namespaced_pod(name=pod_name, namespace=namespace)
        print("Pod definition fetched successfully.")
        
        # 2. Create a new Pod object from the existing one, removing the sidecar
        new_pod_body = client.V1Pod(
            api_version=existing_pod.api_version,
            kind=existing_pod.kind,
            metadata=client.V1ObjectMeta(
                name=existing_pod.metadata.name,
                namespace=existing_pod.metadata.namespace,
                labels=existing_pod.metadata.labels,
                annotations=existing_pod.metadata.annotations,
                # Clear resourceVersion and uid to avoid conflicts when creating a new pod
                resource_version=None,
                uid=None
            ),
            spec=client.V1PodSpec(
                # Filter out the diagnostic sidecar container
                containers=[c for c in existing_pod.spec.containers if c.name != DIAGNOSTIC_SIDECAR_NAME],
                # Filter out the diagnostic volume if it exists
                volumes=[v for v in existing_pod.spec.volumes if v.name != DIAGNOSTIC_VOLUME_NAME] if existing_pod.spec.volumes else None,
                service_account_name=existing_pod.spec.service_account_name,
                node_name=existing_pod.spec.node_name,
                affinity=existing_pod.spec.affinity,
                tolerations=existing_pod.spec.tolerations,
                # Set share_process_namespace back to original value or None
                share_process_namespace=False
            )
        )
        
        # 3. Delete the Pod with the sidecar
        print(f"Deleting Pod '{pod_name}' with diagnostic sidecar... (This will cause brief downtime)")
        api_client.delete_namespaced_pod(name=pod_name, namespace=namespace, body=client.V1DeleteOptions())
        
        # Wait for the Pod to be deleted
        wait_for_pod_deletion(api_client, pod_name, namespace)
        print(f"Pod '{pod_name}' with diagnostic sidecar deleted.")
        
        # 4. Create the new Pod without the sidecar
        print(f"Creating new Pod '{pod_name}' without diagnostic sidecar...")
        created_pod = api_client.create_namespaced_pod(body=new_pod_body, namespace=namespace)
        print(f"New Pod '{created_pod.metadata.name}' created successfully.")
        
        # Wait for the new Pod to be ready
        wait_for_pod_ready(api_client, created_pod.metadata.name, namespace)
        print(f"New Pod '{created_pod.metadata.name}' is Running and Ready.")
        print("Diagnostic sidecar removal complete. Pod has been restored to its original state.")
        
    except ApiException as e:
        print(f"Kubernetes API Error during sidecar removal: {e}")
        if e.status == 404:
            print(f"Error: Pod '{pod_name}' not found in namespace '{namespace}'.")
        elif e.status == 409: # Conflict, usually means resourceVersion issue if not cleared
            print(f"Error: Conflict when creating Pod. Check if resourceVersion was cleared correctly. {e}")
        else:
            print(f"An unexpected API error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during sidecar removal: {e}")

# --- Check Permissions ---
def check_permissions(api_client, namespace):
    """
    Check if the current user has the necessary permissions to create and delete pods in the namespace.
    """
    print(f"\n--- Checking permissions in namespace: {namespace} ---")
    try:
        # Try to list pods in the namespace to check read permissions
        api_client.list_namespaced_pod(namespace=namespace, limit=1)
        print(f"✓ You have permission to list pods in namespace '{namespace}'")
        
        # Unfortunately, there's no direct way to check if you have permission to create/delete pods
        # without actually trying to do it. We'll have to rely on the actual operation to check.
        print(f"Note: Full permissions to create/delete pods can only be verified during the actual operation.")
        return True
    except ApiException as e:
        if e.status == 403:  # Forbidden
            print(f"✗ Permission denied: You don't have sufficient permissions in namespace '{namespace}'")
            print(f"Error details: {e}")
            return False
        elif e.status == 404:  # Not Found
            print(f"✗ Namespace '{namespace}' not found. Please check if the namespace exists.")
            return False
        else:
            print(f"✗ Error checking permissions: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error checking permissions: {e}")
        return False

# --- Check Pod Status ---
def check_pod_exists(api_client, pod_name, namespace):
    """
    Check if the pod exists and is in a running state.
    """
    print(f"\n--- Checking if pod exists: {pod_name} in namespace: {namespace} ---")
    try:
        pod = api_client.read_namespaced_pod(name=pod_name, namespace=namespace)
        print(f"✓ Pod '{pod_name}' found in namespace '{namespace}'")
        print(f"  Status: {pod.status.phase}")
        
        # Check if the pod is controlled by a higher-level resource
        if pod.metadata.owner_references:
            for owner in pod.metadata.owner_references:
                print(f"  ⚠ Warning: This pod is managed by {owner.kind} '{owner.name}'")
                print(f"    Modifying this pod directly may cause conflicts with the controller.")
                print(f"    The controller may recreate or replace the pod, undoing your changes.")
        
        return pod
    except ApiException as e:
        if e.status == 404:  # Not Found
            print(f"✗ Pod '{pod_name}' not found in namespace '{namespace}'")
            return None
        else:
            print(f"✗ Error checking pod: {e}")
            return None
    except Exception as e:
        print(f"✗ Unexpected error checking pod: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure you have a Pod named 'my-troubleshooted-app' running in the 'default' namespace
    # before running this script. You can use the YAML from the previous response to create one.

    # Example: Create a dummy Nginx pod for testing if you don't have one
    # kubectl run my-troubleshooted-app --image=nginx --restart=Never --port=80

    api = initialize_k8s_client()
    
    # First check if we have the necessary permissions
    if not check_permissions(api, TARGET_NAMESPACE):
        print("\n⚠ WARNING: You may not have sufficient permissions to perform all operations.")
        print("  Proceeding anyway, but some operations may fail.")
    
    # Then check if the pod exists
    pod_check = check_pod_exists(api, TARGET_POD_NAME, TARGET_NAMESPACE)
    if pod_check is None:
        print(f"\n✗ Cannot proceed: Pod '{TARGET_POD_NAME}' not found in namespace '{TARGET_NAMESPACE}'")
        print("  Please check the pod name and namespace and try again.")
        exit(1)
    
    # Inject the sidecar and check if it was actually injected (not already present)
    pod = inject_sidecar_to_pod(api, TARGET_POD_NAME, TARGET_NAMESPACE)
    
    # Only proceed with diagnostics if the sidecar was actually injected or already exists
    if pod is None:
        # New pod was created with sidecar, proceed with diagnostics
        print("\n✓ Sidecar was successfully injected. Proceeding with diagnostics.")
        run_diagnostic_sidecar(api, TARGET_POD_NAME, TARGET_NAMESPACE)
        download_diagnostic_data(api, TARGET_POD_NAME, TARGET_NAMESPACE)    
        remove_sidecar_from_pod(api, TARGET_POD_NAME, TARGET_NAMESPACE)
    else:
        # Pod already had the sidecar, just run diagnostics without removal
        print("\n✓ Using existing diagnostic sidecar. Proceeding with diagnostics.")
        run_diagnostic_sidecar(api, TARGET_POD_NAME, TARGET_NAMESPACE)
        download_diagnostic_data(api, TARGET_POD_NAME, TARGET_NAMESPACE)

    print("\n--- Next Steps ---")
    print(f"After successful injection, you can execute commands in the '{DIAGNOSTIC_SIDECAR_NAME}' sidecar:")
    print(f"kubectl exec -it {TARGET_POD_NAME} -c {DIAGNOSTIC_SIDECAR_NAME} -- tcpdump -i eth0 -w {DIAGNOSTIC_VOLUME_MOUNT_PATH}/capture.pcap -G 30 -W 1")
    print(f"To copy the pcap file: kubectl cp {TARGET_POD_NAME}:{DIAGNOSTIC_VOLUME_MOUNT_PATH}/capture.pcap ./capture.pcap -c {DIAGNOSTIC_SIDECAR_NAME}")
    print(f"To view logs from the sidecar: kubectl logs {TARGET_POD_NAME} -c {DIAGNOSTIC_SIDECAR_NAME}")