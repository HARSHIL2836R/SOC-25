@echo off
echo Starting Research Paper Analysis Agent with LangGraph...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please copy .env.example to .env and add your API keys.
    echo.
    echo Required API Keys:
    echo - GROQ_API_KEY (required for LLM functionality)
    echo - TAVILY_API_KEY (optional, for internet search)
    echo.
    pause
    exit /b 1
)

REM Test the LangGraph agent
echo.
echo Testing LangGraph agent setup...
python test_langgraph_agent.py

echo.
echo Starting Streamlit application...
streamlit run app.py

pause
