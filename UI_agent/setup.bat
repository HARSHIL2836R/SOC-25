@echo off
echo 🔧 Research Agent Setup Helper
echo ==============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Check if requirements are installed
echo.
echo 📦 Checking dependencies...
python -c "import streamlit, tavily, langgraph" >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies already installed
)

REM Check if .env file exists
echo.
echo 🔍 Checking configuration...
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env >nul
    echo ✅ .env file created
) else (
    echo ✅ .env file exists
)

REM Run configuration helper
echo.
echo 🔑 Running configuration helper...
python setup_config.py

REM Test setup
echo.
echo 🧪 Testing configuration...
python test_setup.py

echo.
echo 🚀 Setup complete! You can now run:
echo    streamlit run app.py
echo.
pause
