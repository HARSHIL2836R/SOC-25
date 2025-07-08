#!/usr/bin/env python3
"""
Test script to verify the modular structure works correctly.
"""

from typing import Any

def test_imports() -> bool:
    """Test that all modules can be imported without errors."""
    try:
        print("Testing modular imports...")
        
        import config
        print("✅ config.py imported successfully")
        print(f"   - App title: {config.APP_TITLE}")
        
        import ui_components
        print("✅ ui_components.py imported successfully")
        
        import llm_utils
        print("✅ llm_utils.py imported successfully")
        
        import pdf_processor
        print("✅ pdf_processor.py imported successfully")
        
        import vector_store
        print("✅ vector_store.py imported successfully")
        
        import research_chain
        print("✅ research_chain.py imported successfully")
        
        print("\n🎉 All modules imported successfully!")
        print("🎉 Modular structure is working correctly!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_functions() -> bool:
    """Test that key functions are accessible."""
    try:
        from config import ensure_papers_folder, SAMPLE_PAPERS
        from pdf_processor import get_saved_papers
        
        print("\nTesting key functions...")
        
        # Test config functions
        papers_folder = ensure_papers_folder()
        print(f"✅ Papers folder: {papers_folder}")
        print(f"✅ Sample papers count: {len(SAMPLE_PAPERS)}")
        
        # Test pdf_processor functions
        saved_papers = get_saved_papers()
        print(f"✅ Saved papers count: {len(saved_papers)}")
        
        print("🎉 Key functions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Function test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MODULAR STRUCTURE TEST")
    print("=" * 50)
    
    success: bool = True
    success &= test_imports()
    success &= test_functions()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 You can now run: streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please check the error messages above")
    print("=" * 50)
