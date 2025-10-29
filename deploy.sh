#!/bin/bash

# AI Code Detection System Deployment Script

set -e

echo "🚀 Starting AI Code Detection System Deployment"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version is not supported. Please install Python $required_version or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed data/train data/test data/validation
mkdir -p models results logs web_app/static web_app/templates

# Create placeholder files
touch data/raw/.gitkeep data/processed/.gitkeep data/train/.gitkeep data/test/.gitkeep data/validation/.gitkeep
touch models/.gitkeep results/.gitkeep logs/.gitkeep

# Run basic tests
echo "🧪 Running basic tests..."
python -m pytest tests/test_basic.py -v

# Train models (optional)
if [ "$1" = "--train" ]; then
    echo "🤖 Training models..."
    python main.py
else
    echo "⏭️ Skipping model training (use --train to train models)"
fi

# Start the application
echo "🌐 Starting web application..."
echo "📱 Open your browser to http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"

streamlit run web_app/app.py --server.port=8501 --server.address=0.0.0.0
