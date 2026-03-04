#!/bin/bash

# Hierarchical RAG Setup Script
# This script automates the setup process

echo "=========================================="
echo "Hierarchical RAG Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy language model..."
read -p "Do you want to download spaCy model for better entity extraction? (y/n): " install_spacy

if [[ $install_spacy == "y" || $install_spacy == "Y" ]]; then
    python -m spacy download en_core_web_sm
    echo "spaCy model downloaded successfully!"
else
    echo "Skipping spaCy model. You can install it later with:"
    echo "  python -m spacy download en_core_web_sm"
fi

# Create .env file
echo ""
read -p "Do you want to create a .env file for your API key? (y/n): " create_env

if [[ $create_env == "y" || $create_env == "Y" ]]; then
    read -p "Enter your Google Gemini API key: " api_key
    echo "GOOGLE_API_KEY=$api_key" > .env
    echo ".env file created!"
else
    echo "Skipping .env file. You can enter your API key in the app sidebar."
fi

# Create sample documents directory
echo ""
echo "Creating sample documents directory..."
mkdir -p sample_documents

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment (if not already active):"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     source venv/Scripts/activate"
else
    echo "     source venv/bin/activate"
fi
echo "  2. Run the app:"
echo "     streamlit run app_hierarchical.py"
echo ""
echo "Happy analyzing! 📄⚖️"
