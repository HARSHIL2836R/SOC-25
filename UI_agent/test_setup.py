#!/usr/bin/env python3
"""
Test script for checking API configurations and internet search functionality.
Run this script to verify your setup before using the main application.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing package imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False
    
    try:
        from tavily import TavilyClient
        print("✅ Tavily imported successfully")
    except ImportError:
        print("❌ Tavily not found. Install with: pip install tavily-python")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ Python-dotenv imported successfully")
    except ImportError:
        print("❌ Python-dotenv not found. Install with: pip install python-dotenv")
        return False
    
    try:
        import langgraph
        print("✅ LangGraph imported successfully")
    except ImportError:
        print("❌ LangGraph not found. Install with: pip install langgraph")
        return False
    
    return True

def test_environment():
    """Test environment variables"""
    print("\n🔧 Testing environment configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("❌ GROQ_API_KEY not configured")
        print("   Get your key from: https://console.groq.com/keys")
        print("   Add it to your .env file")
        return False
    else:
        print("✅ GROQ_API_KEY configured")
    
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        print("⚠️  TAVILY_API_KEY not configured (internet search will be disabled)")
        print("   Get your key from: https://tavily.com/")
        print("   Add it to your .env file for internet search functionality")
    else:
        print("✅ TAVILY_API_KEY configured")
        return test_tavily_connection(tavily_key)
    
    return True

def test_tavily_connection(api_key):
    """Test Tavily API connection"""
    print("\n🌐 Testing Tavily API connection...")
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        # Test with a simple search
        result = client.search("machine learning", max_results=1)
        
        if result and result.get('results'):
            print("✅ Tavily API connection successful")
            print(f"   Found {len(result['results'])} result(s)")
            return True
        else:
            print("⚠️  Tavily API connected but no results returned")
            return True
            
    except Exception as e:
        print(f"❌ Tavily API connection failed: {e}")
        print("   Check your API key and internet connection")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "config.py",
        "research_chain.py",
        "llm_utils.py",
        "vector_store.py",
        "pdf_processor.py",
        "ui_components.py",
        "app.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Research Agent Configuration Test\n")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test environment
    if not test_environment():
        all_tests_passed = False
    
    # Test file structure
    if not test_file_structure():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    
    if all_tests_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nYou can now run the main application with:")
        print("   streamlit run app.py")
    else:
        print("🔧 Some issues found. Please fix them before running the application.")
        print("\nFor help, check:")
        print("   - README.md for setup instructions")
        print("   - requirements.txt for package dependencies")
        print("   - .env.example for environment variable examples")

if __name__ == "__main__":
    main()
