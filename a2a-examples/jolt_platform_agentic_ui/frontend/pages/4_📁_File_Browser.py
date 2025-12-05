"""
GDrive File Browser Page

This page allows users to:
- Browse files in the mock Google Drive storage
- Upload new JSON files
- View file contents
- Delete files
- Select files to use in the workflow
"""

import streamlit as st
import requests
import json
import os

# Page config
st.set_page_config(
    page_title="File Browser - Jolt Platform",
    page_icon="📁",
    layout="wide"
)

# Custom CSS for file browser
st.markdown("""
<style>
.file-item {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    transition: background-color 0.2s;
}
.file-item:hover {
    background-color: rgba(100, 100, 255, 0.1);
}
.file-icon {
    font-size: 1.5em;
    margin-right: 10px;
}
.file-path {
    color: #888;
    font-size: 0.9em;
}
.upload-zone {
    border: 2px dashed #4CAF50;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    background-color: rgba(76, 175, 80, 0.05);
}
.selected-file {
    background-color: rgba(76, 175, 80, 0.2);
    border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "orchestrator_url" not in st.session_state:
    st.session_state.orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8088")

if "auth_token" not in st.session_state:
    st.session_state.auth_token = "valid_token"

if "selected_input_file" not in st.session_state:
    st.session_state.selected_input_file = None

if "selected_output_file" not in st.session_state:
    st.session_state.selected_output_file = None

# Page title
st.title("📁 GDrive File Browser")
st.markdown("Browse, upload, and manage JSON files for Jolt transformations")

# Authentication check - use 'logged_in' to match Auth page
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Please authenticate first on the Auth page.")
    st.page_link("pages/1_🔐_Auth.py", label="Go to Authentication", icon="🔐")
    st.stop()


def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_files(folder=""):
    """Fetch files from the server"""
    try:
        response = requests.get(
            f"{st.session_state.orchestrator_url}/files/list",
            params={"folder": folder, "auth_token": st.session_state.auth_token}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def read_file(path):
    """Read file content from the server"""
    try:
        response = requests.get(
            f"{st.session_state.orchestrator_url}/files/read",
            params={"path": path, "auth_token": st.session_state.auth_token}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def write_file(path, content):
    """Write file to the server"""
    try:
        response = requests.post(
            f"{st.session_state.orchestrator_url}/files/write",
            json={
                "path": path,
                "content": content,
                "auth_token": st.session_state.auth_token
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def delete_file(path):
    """Delete file from the server"""
    try:
        response = requests.delete(
            f"{st.session_state.orchestrator_url}/files/delete",
            params={"path": path, "auth_token": st.session_state.auth_token}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


# Main layout with tabs
tab1, tab2, tab3 = st.tabs(["📂 Browse Files", "📤 Upload Files", "🎯 Selected Files"])

with tab1:
    st.subheader("Browse Files")
    
    # Refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Fetch and display files
    with st.spinner("Loading files..."):
        result = list_files()
    
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        files = result.get("files", [])
        folder = result.get("folder", "/")
        
        st.caption(f"📍 Current folder: `{folder}`")
        
        if not files:
            st.info("📭 No files found. Upload some files to get started!")
        else:
            # Display files in a grid
            for file_info in files:
                col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1.5, 1, 1])
                
                icon = "📁" if file_info["is_directory"] else "📄"
                
                with col1:
                    st.write(icon)
                
                with col2:
                    st.write(f"**{file_info['name']}**")
                    st.caption(f"`{file_info['path']}`")
                
                with col3:
                    if not file_info["is_directory"]:
                        st.caption(format_file_size(file_info["size"]))
                
                with col4:
                    if not file_info["is_directory"]:
                        if st.button("👁️ View", key=f"view_{file_info['path']}", use_container_width=True):
                            st.session_state[f"viewing_{file_info['path']}"] = True
                
                with col5:
                    if not file_info["is_directory"]:
                        # Selection dropdown
                        select_option = st.selectbox(
                            "Use as",
                            ["—", "Input JSON", "Output JSON"],
                            key=f"select_{file_info['path']}",
                            label_visibility="collapsed"
                        )
                        if select_option == "Input JSON":
                            st.session_state.selected_input_file = file_info['path']
                            st.toast(f"✅ Selected as Input: {file_info['name']}")
                        elif select_option == "Output JSON":
                            st.session_state.selected_output_file = file_info['path']
                            st.toast(f"✅ Selected as Output: {file_info['name']}")
                
                # Show file content if viewing
                if st.session_state.get(f"viewing_{file_info['path']}", False):
                    with st.expander(f"📄 Contents of {file_info['name']}", expanded=True):
                        file_content = read_file(file_info['path'])
                        if "error" in file_content:
                            st.error(file_content["error"])
                        else:
                            content = file_content.get("content", "")
                            try:
                                # Try to parse and pretty-print JSON
                                parsed = json.loads(content)
                                st.code(json.dumps(parsed, indent=2), language="json")
                            except json.JSONDecodeError:
                                st.code(content)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("❌ Close", key=f"close_{file_info['path']}"):
                                st.session_state[f"viewing_{file_info['path']}"] = False
                                st.rerun()
                        with col_b:
                            if st.button("🗑️ Delete", key=f"delete_{file_info['path']}", type="secondary"):
                                result = delete_file(file_info['path'])
                                if result.get("success"):
                                    st.success(f"✅ Deleted {file_info['name']}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete: {result.get('error', 'Unknown error')}")
                
                st.divider()


with tab2:
    st.subheader("Upload Files")
    
    st.markdown("""
    Upload JSON files to use as input/output for Jolt transformations.
    You can either:
    - **Upload a file** from your computer
    - **Paste JSON content** directly
    """)
    
    upload_method = st.radio(
        "Upload Method",
        ["📎 Upload File", "📝 Paste JSON"],
        horizontal=True
    )
    
    if upload_method == "📎 Upload File":
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=["json"],
            help="Upload a JSON file to the mock GDrive storage"
        )
        
        if uploaded_file is not None:
            # Read and preview content
            content = uploaded_file.read().decode("utf-8")
            
            st.markdown("**Preview:**")
            try:
                parsed = json.loads(content)
                st.code(json.dumps(parsed, indent=2), language="json")
                
                # File name input
                new_filename = st.text_input(
                    "Save as",
                    value=uploaded_file.name,
                    help="Enter the filename to save as"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Save to GDrive", type="primary", use_container_width=True):
                        result = write_file(new_filename, content)
                        if result.get("success"):
                            st.success(f"✅ File saved as `{new_filename}`")
                            st.balloons()
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                with col2:
                    use_as = st.selectbox(
                        "Use file as",
                        ["Don't select", "Input JSON", "Output JSON"]
                    )
                    if use_as == "Input JSON":
                        st.session_state.selected_input_file = new_filename
                    elif use_as == "Output JSON":
                        st.session_state.selected_output_file = new_filename
                        
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")
    
    else:  # Paste JSON
        st.markdown("**Paste your JSON content:**")
        
        json_content = st.text_area(
            "JSON Content",
            height=300,
            placeholder='{\n  "example": "Paste your JSON here"\n}'
        )
        
        filename = st.text_input(
            "Filename",
            placeholder="my_file.json",
            help="Enter a filename (must end with .json)"
        )
        
        if json_content and filename:
            # Validate JSON
            try:
                parsed = json.loads(json_content)
                st.success("✅ Valid JSON")
                
                # Ensure .json extension
                if not filename.endswith(".json"):
                    filename = filename + ".json"
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Save to GDrive", type="primary", use_container_width=True):
                        # Pretty-print before saving
                        formatted_content = json.dumps(parsed, indent=2)
                        result = write_file(filename, formatted_content)
                        if result.get("success"):
                            st.success(f"✅ File saved as `{filename}`")
                            st.balloons()
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                with col2:
                    use_as = st.selectbox(
                        "Use file as",
                        ["Don't select", "Input JSON", "Output JSON"],
                        key="paste_use_as"
                    )
                    if use_as == "Input JSON":
                        st.session_state.selected_input_file = filename
                    elif use_as == "Output JSON":
                        st.session_state.selected_output_file = filename
                        
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")


with tab3:
    st.subheader("Selected Files for Workflow")
    
    st.markdown("""
    These are the files currently selected to use in the Jolt transformation workflow.
    You can change your selection in the Browse tab or go directly to the Workflow page.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Input JSON")
        if st.session_state.selected_input_file:
            st.success(f"**Selected:** `{st.session_state.selected_input_file}`")
            
            # Show preview
            file_content = read_file(st.session_state.selected_input_file)
            if "error" not in file_content:
                with st.expander("Preview Content"):
                    try:
                        parsed = json.loads(file_content.get("content", ""))
                        st.code(json.dumps(parsed, indent=2), language="json")
                    except:
                        st.code(file_content.get("content", ""))
            
            if st.button("❌ Clear Input Selection"):
                st.session_state.selected_input_file = None
                st.rerun()
        else:
            st.info("No input file selected")
    
    with col2:
        st.markdown("### 📤 Output JSON (Expected)")
        if st.session_state.selected_output_file:
            st.success(f"**Selected:** `{st.session_state.selected_output_file}`")
            
            # Show preview
            file_content = read_file(st.session_state.selected_output_file)
            if "error" not in file_content:
                with st.expander("Preview Content"):
                    try:
                        parsed = json.loads(file_content.get("content", ""))
                        st.code(json.dumps(parsed, indent=2), language="json")
                    except:
                        st.code(file_content.get("content", ""))
            
            if st.button("❌ Clear Output Selection"):
                st.session_state.selected_output_file = None
                st.rerun()
        else:
            st.info("No output file selected")
    
    st.divider()
    
    # Quick action to go to workflow
    if st.session_state.selected_input_file and st.session_state.selected_output_file:
        st.success("✅ Both files selected! You can now run the workflow.")
        st.page_link(
            "pages/2_🚀_Workflow.py", 
            label="🚀 Go to Workflow Page", 
            icon="🚀",
            use_container_width=True
        )
    else:
        st.warning("⚠️ Please select both input and output files to run the workflow.")
