@echo off
REM AI Code Detection System Deployment Script for Windows

echo 🚀 Starting AI Code Detection System Deployment

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\train" mkdir data\train
if not exist "data\test" mkdir data\test
if not exist "data\validation" mkdir data\validation
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "web_app\static" mkdir web_app\static
if not exist "web_app\templates" mkdir web_app\templates

REM Create placeholder files
echo. > data\raw\.gitkeep
echo. > data\processed\.gitkeep
echo. > data\train\.gitkeep
echo. > data\test\.gitkeep
echo. > data\validation\.gitkeep
echo. > models\.gitkeep
echo. > results\.gitkeep
echo. > logs\.gitkeep

REM Run basic tests
echo 🧪 Running basic tests...
python -m pytest tests\test_basic.py -v

REM Train models (optional)
if "%1"=="--train" (
    echo 🤖 Training models...
    python main.py
) else (
    echo ⏭️ Skipping model training (use --train to train models)
)

REM Start the application
echo 🌐 Starting web application...
echo 📱 Open your browser to http://localhost:8501
echo 🛑 Press Ctrl+C to stop the application

streamlit run web_app\app.py --server.port=8501 --server.address=0.0.0.0

pause
