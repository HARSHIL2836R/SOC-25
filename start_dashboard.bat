@echo off
echo ====================================================
echo Research Paper Analysis Agent - Quick Start
echo ====================================================
echo.

:: Check if .env file exists
if not exist .env (
    echo Creating .env file template...
    echo # Research Paper Analysis Agent - API Keys > .env
    echo # Get your free API key from: https://console.groq.com/keys >> .env
    echo GROQ_API_KEY=your_groq_api_key_here >> .env
    echo.
    echo ⚠️  IMPORTANT: Please edit .env file and add your Groq API key!
    echo    Get it from: https://console.groq.com/keys
    echo.
    pause
)

echo Starting Streamlit dashboard...
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.

c:/Users/Dell/Documents/IIT_academics/SOC@25/.venv.bak/Scripts/streamlit.exe run research_dashboard.py

echo.
echo Dashboard stopped.
pause
