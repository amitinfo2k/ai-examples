# Create project directory
uv init mcp-client
cd mcp-client

# Create virtual environment
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install required packages
uv add mcp anthropic python-dotenv

# Remove boilerplate files
# On Windows:
del main.py
# On Unix or MacOS:
rm main.py

# Create our main file
touch client.py


# Create .env file
touch .env

# Add your key to the .env file
ANTHROPIC_API_KEY=<your key here>

# Add .env to your .gitignore:
echo ".env" >> .gitignore