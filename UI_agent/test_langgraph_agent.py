"""
Test script for the LangGraph-based research agent.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add the UI_agent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from research_chain import ResearchAgent, setup_research_chain
from config import GROQ_API_KEY, TAVILY_API_KEY
from llm_utils import initialize_llm_and_embeddings

def test_agent_setup():
    """Test basic agent setup"""
    print("🔧 Testing LangGraph Agent Setup...")
    
    # Check API keys
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in environment")
        return False
    
    print("✅ GROQ_API_KEY found")
    
    if not TAVILY_API_KEY:
        print("⚠️  TAVILY_API_KEY not found - internet search will be disabled")
    else:
        print("✅ TAVILY_API_KEY found")
    
    return True

def test_llm_initialization():
    """Test LLM and embeddings initialization"""
    print("\n🤖 Testing LLM Initialization...")
    
    try:
        llm, embeddings = initialize_llm_and_embeddings()
        if llm is None or embeddings is None:
            print("❌ Failed to initialize LLM or embeddings")
            return False, None, None
        
        print("✅ LLM and embeddings initialized successfully")
        return True, llm, embeddings
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return False, None, None

def create_mock_vector_store():
    """Create a mock vector store for testing"""
    print("\n📚 Creating mock vector store...")
    
    class MockRetriever:
        def invoke(self, query: str):
            return [
                type('MockDoc', (), {
                    'page_content': f"This is mock content related to: {query}",
                    'metadata': {'section': 'Introduction'}
                })()
            ]
    
    class MockVectorStore:
        def as_retriever(self, **kwargs):
            return MockRetriever()
    
    return MockVectorStore()

def test_agent_creation(llm, embeddings):
    """Test agent creation with mock data"""
    print("\n🤖 Testing Agent Creation...")
    
    try:
        # Create mock vector store and metadata
        vector_store = create_mock_vector_store()
        metadata = {
            'title': 'Test Research Paper',
            'authors': 'Test Author et al.'
        }
        
        # Create agent
        agent = setup_research_chain(vector_store, llm, metadata)
        
        if agent is None:
            print("❌ Failed to create research agent")
            return False, None
        
        print("✅ Research agent created successfully")
        return True, agent
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return False, None

def test_basic_question(agent):
    """Test basic question processing"""
    print("\n💬 Testing Basic Question Processing...")
    
    try:
        question = "What is the main contribution of this paper?"
        response = agent.process_question(question)
        
        if not response or "error" in response.lower():
            print(f"❌ Basic question failed: {response}")
            return False
        
        print("✅ Basic question processed successfully")
        print(f"📝 Response preview: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error processing basic question: {e}")
        return False

def test_search_question(agent):
    """Test question that should trigger internet search"""
    print("\n🔍 Testing Internet Search Question...")
    
    try:
        question = "Are there similar papers to this research?"
        response = agent.process_question(question)
        
        if not response:
            print("❌ Search question failed")
            return False
        
        print("✅ Search question processed successfully")
        print(f"📝 Response preview: {response[:100]}...")
        
        # Check if search was actually performed
        # Note: This might fail if TAVILY_API_KEY is not available
        if "search" in response.lower() or "similar" in response.lower():
            print("🔍 Internet search functionality appears to be working")
        else:
            print("⚠️  Internet search might not have been triggered")
        
        return True
    except Exception as e:
        print(f"❌ Error processing search question: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 LangGraph Research Agent Test Suite")
    print("=" * 50)
    
    # Test 1: Setup
    if not test_agent_setup():
        print("\n❌ Setup tests failed. Check your .env file.")
        return
    
    # Test 2: LLM initialization
    success, llm, embeddings = test_llm_initialization()
    if not success:
        print("\n❌ LLM initialization failed. Check your API keys.")
        return
    
    # Test 3: Agent creation
    success, agent = test_agent_creation(llm, embeddings)
    if not success:
        print("\n❌ Agent creation failed.")
        return
    
    # Test 4: Basic question
    if not test_basic_question(agent):
        print("\n❌ Basic question test failed.")
        return
    
    # Test 5: Search question
    if not test_search_question(agent):
        print("\n⚠️  Search question test had issues (may be due to missing Tavily API key).")
    
    print("\n" + "=" * 50)
    print("🎉 Test Suite Completed!")
    print("✅ Your LangGraph agent is ready to use!")
    print("\n💡 Next steps:")
    print("   1. Run 'streamlit run app.py' to start the application")
    print("   2. Upload a research paper to test with real content")
    print("   3. Try asking questions about similar papers")

if __name__ == "__main__":
    main()
