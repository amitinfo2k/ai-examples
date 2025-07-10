# Create a new directory for our project
uv init k8s-lister
cd k8s-lister

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" kubernetes tabulate

# Create our server file
touch main.py

# Run the server
uv run main.py
