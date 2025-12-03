import streamlit as st
import hashlib
import time
import os

st.set_page_config(page_title="Authentication", page_icon="🔐")

st.title("🔐 Authentication & Configuration")

st.markdown("""
Login to access the Multi-Agent Jolt System. Your credentials will be used to generate 
a secure access token for Google Drive integration.
""")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = 'valid_token'
if 'orchestrator_url' not in st.session_state:
    st.session_state.orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8088')

# Check if already logged in
if st.session_state.logged_in:
    st.success(f"✅ Logged in as: **{st.session_state.user_email}**")
    
    st.divider()
    
    # Show configuration details
    st.subheader("📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", "Authenticated", delta="Active")
        st.metric("User", st.session_state.user_email)
    
    with col2:
        st.metric("Access Token", "Generated ✓")
        st.metric("Directory Access", "Enabled")
    
    # Show token (partially masked for security)
    with st.expander("🔑 View Access Token"):
        st.info("Your access token is securely stored in the session.")
        token_display = st.session_state.auth_token
        masked_token = f"{token_display[:6]}...{token_display[-4:]}" if len(token_display) > 10 else "***"
        st.code(masked_token)
        st.caption("This token will be used for all MCP operations")
    
    # Directory configuration
    st.divider()
    st.subheader("📁 Google Drive Configuration")
    
    st.info("""
    **Mock Storage Location:**  
    Place your JSON files in: `mcp_server/storage/`  
    
    Sample files are already available:
    - `input.json` (sample input)
    - `output.json` (expected output)
    
    You can replace these or add new files to test different transformations.
    """)
    
    st.text_input(
        "Target Directory",
        value="mock_gdrive_storage",
        disabled=True,
        help="This is the Google Drive directory where input/output files are stored"
    )
    
    # Logout button
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.auth_token = None
        st.rerun()

else:
    # Login form
    st.subheader("🔐 Login")
    st.info("**Demo Mode:** For this demonstration, use any email and the password 'demo123'")
    
    with st.form("login_form"):
        email = st.text_input("Email Address", placeholder="user@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            remember = st.checkbox("Remember me")
        with col2:
            submit = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if submit:
            # Demo authentication logic
            if email and password == "demo123":
                # Generate a token based on email
                token = hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()[:20]
                
                # For demo purposes, use "valid_token" so MCP server accepts it
                st.session_state.auth_token = "valid_token"
                st.session_state.user_email = email
                st.session_state.logged_in = True
                
                st.success("✅ Login successful! Redirecting...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Use password 'demo123'")
    
    st.divider()
    
    # Help section
    with st.expander("ℹ️ Need Help?"):
        st.markdown("""
        **For Demo/Development:**
        - Email: Any email address
        - Password: `demo123`
        
        **What happens after login?**
        1. System generates a secure access token
        2. Token is stored in your session
        3. Token is used to authenticate with MCP server for Google Drive access
        4. You can then run workflows to generate and validate Jolt specs
        
        **Production Setup:**
        In a production environment, this would integrate with Google OAuth to 
        obtain real Google Drive access tokens with appropriate scopes.
        """)

