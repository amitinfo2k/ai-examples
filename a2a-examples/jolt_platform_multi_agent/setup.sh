#!/bin/bash

# JOLT Multi-Agent Platform Setup Script

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║     JOLT Multi-Agent Platform - Setup Script            ║"
echo "║                                                          ║"
echo "║   🤖 CrewAI   +   ✅ LangChain                          ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo ""
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt -q
if [ $? -eq 0 ]; then
    echo "   ✅ All dependencies installed successfully"
else
    echo "   ❌ Error installing dependencies"
    exit 1
fi

# Setup .env file
echo ""
echo "🔐 Setting up environment configuration..."
if [ -f ".env" ]; then
    echo "   .env file already exists. Skipping..."
else
    cp .env.example .env
    echo "   ✅ Created .env file from template"
    echo ""
    echo "   ⚠️  IMPORTANT: Please edit .env and add your OpenAI API key!"
    echo "   Run: nano .env"
fi

# Create output directories
echo ""
echo "📁 Creating output directories..."
mkdir -p output
mkdir -p api_output
mkdir -p quickstart_output
echo "   ✅ Output directories created"

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python3 -c "import crewai; import langchain; import fastapi" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ All packages verified successfully"
else
    echo "   ❌ Package verification failed"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║                   ✅ SETUP COMPLETE!                     ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Next Steps:"
echo ""
echo "   1. Configure your API key:"
echo "      nano .env"
echo "      (Add your OPENAI_API_KEY)"
echo ""
echo "   2. Activate the virtual environment (if not already):"
echo "      source venv/bin/activate"
echo ""
echo "   3. Run the quick start demo:"
echo "      python quickstart.py"
echo ""
echo "   4. Or start the API server:"
echo "      python platform/api_server.py"
echo ""
echo "   5. Read the documentation:"
echo "      cat README.md"
echo ""
echo "✨ Happy coding!"
echo ""
