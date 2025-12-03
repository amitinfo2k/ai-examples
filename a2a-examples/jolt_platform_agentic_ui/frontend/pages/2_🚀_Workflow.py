import streamlit as st
import requests
import json
import time
import os
from websockets.sync.client import connect
import threading
import queue

class WebSocketManager:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.messages = queue.Queue()
        self.thread = None
        self.stop_event = threading.Event()
    
    def connect(self, url):
        if self.connected:
            self.disconnect()
        
        self._connect_url = url  # Store URL for error reporting
            
        def run_ws():
            last_ping_time = time.time()
            try:
                # Add connection timeout of 10 seconds
                self.ws = connect(url, open_timeout=10)
                self.connected = True
                
                while not self.stop_event.is_set():
                    try:
                        # Send a ping every 5 seconds to keep the connection alive
                        if time.time() - last_ping_time > 5:
                            self.ws.send("ping")
                            last_ping_time = time.time()

                        # Use a shorter timeout to remain responsive to the stop event
                        message = self.ws.recv(timeout=1)
                        if message:
                            if message == "pong": # Server acknowledged our ping
                                continue
                            try:
                                self.messages.put(json.loads(message))
                            except json.JSONDecodeError as je:
                                self.messages.put({
                                    "type": "error", 
                                    "message": f"Invalid JSON received: {str(je)}"
                                })
                    except TimeoutError:
                        # This is expected if no message is received, continue loop
                        continue
                    except Exception as e:
                        if not self.stop_event.is_set():
                            self.messages.put({
                                "type": "error", 
                                "message": f"WebSocket receive error: {str(e)}"
                            })
                            break
            except Exception as e:
                error_msg = f"WebSocket connection failed: {str(e)}"
                self.messages.put({"type": "error", "message": error_msg})
                print(error_msg)  # Log the error for debugging
            finally:
                self.connected = False
                if hasattr(self, 'ws') and self.ws:
                    try:
                        self.ws.close()
                    except Exception as close_error:
                        print(f"Error closing WebSocket: {str(close_error)}")
        
        self.thread = threading.Thread(target=run_ws, daemon=True)
        self.thread.start()
        
        # Wait for connection to be established or fail
        start_time = time.time()
        while not self.connected and time.time() - start_time < 10:  # 10 second timeout
            time.sleep(0.1)
        
        if not self.connected:
            print(f"Failed to connect to WebSocket at {self._connect_url} after 10 seconds")
        
        return self.connected
                
    def poll_workflow_status(self, task_id: str, status, progress_bar, status_text, log_container, max_attempts: int = 300):
        """Fallback polling mechanism when WebSocket fails"""
        attempt = 0
        
        while attempt < max_attempts and not self.stop_event.is_set():
            try:
                response = requests.get(
                    f"{st.session_state.orchestrator_url}/workflow/status/{task_id}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    progress = data.get("progress_percentage", 0)
                    current_step = data.get("current_step", "Processing...")
                    logs = data.get("logs", [])
                    task_status = data.get("status")
                    
                    # Update UI
                    progress_bar.progress(progress / 100)
                    status_text.text(f"📍 {current_step} ({progress}%)")
                    log_container.text_area("Recent Logs", "\n".join(logs[-10:]), height=150)
                    
                    # Check if workflow is done
                    if task_status in ["completed", "failed", "needs_review"]:
                        st.session_state.workflow_progress = {
                            "status": task_status,
                            "progress": progress,
                            "current_step": current_step,
                            "logs": logs
                        }
                        
                        if "result" in data:
                            st.session_state.workflow_result = data["result"]
                        
                        st.rerun()
                        return
                        
                attempt += 1
                time.sleep(0.1)
                
            except requests.RequestException as e:
                status.update(label="⚠️ Connection error", state="error")
                log_container.error(f"Failed to get status: {str(e)}")
                break
    
    def disconnect(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.connected = False
        self.stop_event.clear()
    
    def get_messages(self):
        messages = []
        while not self.messages.empty():
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def __del__(self):
        self.disconnect()

# Initialize WebSocket manager in session state
if 'ws_manager' not in st.session_state:
    st.session_state.ws_manager = WebSocketManager()

st.set_page_config(page_title="Workflow", page_icon="🚀", layout="wide")

st.title("🚀 Jolt Spec Workflow")

# Initialize session state
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = 'valid_token'
if 'orchestrator_url' not in st.session_state:
    st.session_state.orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8088')
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Check if authenticated
if 'auth_token' not in st.session_state or not st.session_state.auth_token:
    st.warning("⚠️ Please configure authentication first!")
    st.stop()

st.markdown("""
Trigger the complete workflow: **Generation → Validation → A2A Debugging**
""")

# Info about file placement
st.info("""
📁 **File Location:** Place your JSON files in `mcp_server/storage/` directory.  
Sample files (`input.json`, `output.json`) are already provided. You can replace them or add new ones.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Input Configuration")
    input_file = st.text_input("Input JSON File", value="input.json")
    output_file = st.text_input("Output JSON File", value="output.json")

with col2:
    st.subheader("🎯 Workflow Options")
    st.info("The workflow will automatically generate and validate the Jolt spec")

st.divider()

# --- Workflow Execution ---
def run_workflow():
    # Clear previous results
    st.session_state.pop('workflow_result', None)
    st.session_state.pop('workflow_progress', None)

    # Prepare request data from UI inputs
    request_data = {
        "task_type": "generate_spec",
        "input_file_path": input_file,
        "output_file_path": output_file,
        "auth_mapping": {
            "user_id": "demo_user",
            "gdrive_token": st.session_state.auth_token,
            "target_directory_id": "mock_gdrive_storage"
        },
        "description": "Generate and validate Jolt spec"
    }

    try:
        response = requests.post(
            f"{st.session_state.orchestrator_url}/workflow/generate-and-validate",
            json=request_data,
            timeout=30
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        workflow_info = response.json()
        task_id = workflow_info.get("task_id")
        if not task_id:
            st.error("❌ No task ID returned from workflow")
            return

        # Initialize progress tracking in session state
        st.session_state.workflow_progress = {
            "task_id": task_id,
            "progress": 0,
            "status": "running",
            "current_step": "Initializing...",
            "logs": []
        }
        st.rerun() # Rerun to enter monitoring mode

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout - The server took too long to respond.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to start workflow: {e}")
        try:
            st.json(e.response.json())
        except Exception:
            st.text(e.response.text[:1000] if e.response else "No response text.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")

if st.button("▶️ Run Complete Workflow", type="primary", use_container_width=True):
    run_workflow()

# --- Workflow Monitoring and Results Display ---
if 'workflow_progress' in st.session_state and st.session_state.workflow_progress:
    task_id = st.session_state.workflow_progress.get("task_id")

    # If workflow is done, show final status and skip to results
    if 'workflow_result' in st.session_state and st.session_state.workflow_result:
        final_status = st.session_state.workflow_result.get("status")
        if final_status == "completed":
            st.success("✅ Workflow completed successfully!")
        elif final_status == "failed":
            st.error("❌ Workflow failed!")
        else:
            st.warning("⚠️ Workflow needs review.")
    
    # Otherwise, show the real-time monitoring UI
    else:
        with st.status("📡 Running workflow...", expanded=True) as status:
            # Create UI elements for real-time updates
            progress_bar = st.progress(st.session_state.workflow_progress.get("progress", 0) / 100.0)
            status_text = st.empty()
            log_container = st.empty()

            # Connect to WebSocket if not already connected
            ws_url = st.session_state.orchestrator_url.replace("http", "ws") + f"/ws/status/{task_id}"
            if not st.session_state.ws_manager.connected:
                if not st.session_state.ws_manager.connect(ws_url):
                    st.error("Failed to connect to WebSocket. Please try again.")
                    st.stop()

            start_time = time.time()
            max_runtime = 600  # 10 minutes

            while True:
                if time.time() - start_time > max_runtime:
                    status.update(label="⏱️ Workflow timed out", state="error")
                    st.session_state.ws_manager.disconnect()
                    break

                messages = st.session_state.ws_manager.get_messages()
                if not messages and not st.session_state.ws_manager.connected and 'workflow_result' not in st.session_state:
                    status.update(label="🔌 Connection lost", state="error")
                    break

                for msg in messages:
                    if msg.get("type") == "status_update":
                        # Update state from message
                        progress = msg.get("progress_percentage", st.session_state.workflow_progress["progress"])
                        current_step = msg.get("current_step", st.session_state.workflow_progress["current_step"])
                        logs = msg.get("logs", st.session_state.workflow_progress["logs"])
                        workflow_status = msg.get("status")

                        # Update UI elements directly
                        progress_bar.progress(progress / 100.0)
                        status_text.markdown(f"**Status:** {current_step} (`{progress}%`)")
                        log_container.text_area("Workflow Logs", "\n".join(logs), height=200, key=f"logs_{time.time()}")

                        # Check for completion
                        if workflow_status in ["completed", "failed", "needs_review"]:
                            # Store all logs and progress in the result
                            st.session_state.workflow_result = {
                                "status": workflow_status,
                                "message": f"Workflow finished with status: {workflow_status}",
                                "result": msg.get("result", {}),
                                "logs": logs,  # Store the logs
                                "progress": progress,
                                "current_step": current_step
                            }
                            
                            # Update the UI one last time before disconnecting
                            if workflow_status == "completed":
                                status.update(label="✅ Workflow Completed", state="complete")
                            elif workflow_status == "failed":
                                status.update(label="❌ Workflow Failed", state="error")
                            else:
                                # For 'needs_review' status, use 'error' state but with warning icon
                                status.update(label="⚠️ Workflow Needs Review", state="error")
                            
                            # Ensure logs are visible before disconnecting
                            log_container.text_area(
                                "Workflow Logs", 
                                "\n".join(logs), 
                                height=200, 
                                key=f"final_logs_{time.time()}",
                                disabled=True
                            )
                            
                            st.session_state.ws_manager.disconnect()
                            time.sleep(0.5)  # Small delay to show final status
                            st.rerun()
                            st.stop()

                time.sleep(0.2) # Small delay to prevent high CPU usage


# Display results if available in session state
if 'workflow_result' in st.session_state:
    result = st.session_state.workflow_result
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📝 Jolt Spec", "✅ Validation", "💬 A2A Messages"])
    
    with tab1:
        st.subheader("Workflow Summary")
        
        # Display status with appropriate icon
        status = result.get("status", "Unknown").capitalize()
        if status.lower() == "completed":
            st.success(f"✅ {status}")
        elif status.lower() == "failed":
            st.error(f"❌ {status}")
        elif "needs_review" in status.lower():
            st.warning(f"⚠️ {status}")
        else:
            st.info(f"ℹ️ {status}")
            
        # Display progress and current step if available
        if "progress" in result:
            st.metric("Progress", f"{result['progress']}%")
        if "current_step" in result:
            st.caption(f"Last step: {result['current_step']}")
            
        # Display the workflow logs
        st.subheader("Execution Log")
        if "logs" in result and result["logs"]:
            # Create a text area with the logs that can be scrolled
            st.text_area(
                "Workflow Execution Logs",
                "\n".join(str(log) for log in result["logs"]),
                height=300,
                disabled=True,
                key="workflow_logs_display"
            )
        elif "result" in result and "workflow_log" in result["result"]:
            # Fallback to old log format if available
            st.text_area(
                "Workflow Execution Logs",
                "\n".join(str(log) for log in result["result"]["workflow_log"]),
                height=300,
                disabled=True,
                key="workflow_logs_display_fallback"
            )
        else:
            st.info("No execution logs available.")
    
    with tab2:
        st.subheader("Generated Jolt Specification")
        if "result" in result and "jolt_spec" in result["result"]:
            st.json(result["result"]["jolt_spec"])
        else:
            st.warning("No Jolt spec in response")
    
    with tab3:
        st.subheader("Validation Results")
        if "result" in result and "validation" in result["result"]:
            validation = result["result"]["validation"]
            
            if validation.get("is_valid"):
                st.success("✅ Validation Passed!")
            else:
                st.error("❌ Validation Failed")
                
                if "errors" in validation:
                    st.subheader("Errors Found:")
                    for error in validation["errors"]:
                        st.warning(f"**Path:** {error.get('path', 'N/A')}")
                        st.text(f"Expected: {error.get('expected', 'N/A')}")
                        st.text(f"Actual: {error.get('actual', 'N/A')}")
                        st.text(f"Description: {error.get('error_description', 'N/A')}")
                        st.divider()
            
            if "actual_output" in validation:
                with st.expander("View Actual Output"):
                    st.json(validation["actual_output"])
            # Manual Refinement Section
            if not validation.get("is_valid"):
                st.divider()
                st.subheader("🛠️ Manual Refinement (Human-in-the-Loop)")
                st.info("The automated validation failed. You can manually edit the Jolt spec below and re-validate.")
                
                # Get current spec and expected output from result
                current_spec = result.get("result", {}).get("jolt_spec", [])
                current_expected = result.get("result", {}).get("expected_output", {})
                
                with st.form("manual_refinement_form"):
                    col_edit1, col_edit2 = st.columns(2)
                    
                    with col_edit1:
                        # Text area for editing Jolt Spec
                        edited_spec_str = st.text_area(
                            "Edit Jolt Spec", 
                            value=json.dumps(current_spec, indent=2),
                            height=400
                        )
                    
                    with col_edit2:
                        # Text area for editing Expected Output
                        edited_expected_str = st.text_area(
                            "Edit Expected Output JSON", 
                            value=json.dumps(current_expected, indent=2),
                            height=400,
                            help="If the expected output is wrong, correct it here."
                        )
                    
                    submit_button = st.form_submit_button("🔄 Re-Validate Manual Spec")
                    
                    if submit_button:
                        try:
                            # Parse edited inputs
                            edited_spec = json.loads(edited_spec_str)
                            edited_expected = json.loads(edited_expected_str)
                            
                            # Validate and unwrap if needed
                            # Check if the spec was accidentally wrapped
                            if isinstance(edited_spec, dict):
                                if "jolt_spec" in edited_spec and "expected_output" in edited_spec:
                                    # User accidentally pasted the whole request body
                                    st.warning("Detected wrapped structure, unwrapping jolt_spec...")
                                    edited_spec = edited_spec["jolt_spec"]
                                    edited_expected = edited_spec.get("expected_output", edited_expected)
                            
                            # Ensure spec is a list
                            if isinstance(edited_spec, dict):
                                edited_spec = [edited_spec]
                            
                            with st.spinner("Validating manual spec..."):
                                # Call validate endpoint directly
                                val_response = requests.post(
                                    f"{st.session_state.orchestrator_url}/validate",
                                    params={
                                        "input_path": result.get("result", {}).get("input_file_path", input_file),
                                        "output_path": result.get("result", {}).get("output_file_path", output_file),
                                        "auth_token": st.session_state.auth_token
                                    },
                                    json={
                                        "jolt_spec": edited_spec,
                                        "expected_output": edited_expected
                                    }
                                )
                                
                                if val_response.status_code == 200:
                                    manual_result = val_response.json()
                                    if manual_result.get("is_valid"):
                                        st.success("✅ Manual Fix Worked! Validation Passed.")
                                        st.balloons()
                                        
                                        # Update session state with new valid result
                                        st.session_state.workflow_result["result"]["validation"] = manual_result
                                        st.session_state.workflow_result["result"]["jolt_spec"] = edited_spec
                                        st.session_state.workflow_result["result"]["expected_output"] = edited_expected
                                        
                                        # Force a rerun to update the UI
                                        time.sleep(1) # Give user a moment to see the success message
                                        st.rerun()
                                    else:
                                        st.error("❌ Still Invalid")
                                        st.json(manual_result.get("errors", []))
                                else:
                                    st.error(f"Validation failed: {val_response.text}")
                                    
                        except json.JSONDecodeError:
                            st.error("❌ Invalid JSON format in editor")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    with tab4:
        st.subheader("Agent-to-Agent Communication")
        if "result" in result and "validation" in result["result"] and "a2a_messages" in result["result"]["validation"]:
            messages = result["result"]["validation"]["a2a_messages"]
            
            if messages:
                for i, msg in enumerate(messages):
                    with st.chat_message(msg.get("sender", "unknown")):
                        st.markdown(f"**{msg.get('message_type', 'Unknown Type')}**")
                        st.json(msg.get("content", {}))
            else:
                st.info("No A2A messages (validation passed on first attempt)")
        else:
            st.info("No A2A communication data available")
