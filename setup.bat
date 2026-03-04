@echo off
REM Hierarchical RAG Setup Script for Windows

echo ==========================================
echo Hierarchical RAG Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Download spaCy model
echo.
set /p install_spacy="Do you want to download spaCy model for better entity extraction? (y/n): "

if /i "%install_spacy%"=="y" (
    python -m spacy download en_core_web_sm
    echo spaCy model downloaded successfully!
) else (
    echo Skipping spaCy model. You can install it later with:
    echo   python -m spacy download en_core_web_sm
)

REM Create .env file
echo.
set /p create_env="Do you want to create a .env file for your API key? (y/n): "

if /i "%create_env%"=="y" (
    set /p api_key="Enter your Google Gemini API key: "
    echo GOOGLE_API_KEY=%api_key% > .env
    echo .env file created!
) else (
    echo Skipping .env file. You can enter your API key in the app sidebar.
)

REM Create sample documents directory
echo.
echo Creating sample documents directory...
if not exist "sample_documents" mkdir sample_documents

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To run the application:
echo   1. Activate virtual environment (if not already active):
echo      venv\Scripts\activate.bat
echo   2. Run the app:
echo      streamlit run app_hierarchical.py
echo.
echo Happy analyzing! 📄⚖️
echo.
pause
