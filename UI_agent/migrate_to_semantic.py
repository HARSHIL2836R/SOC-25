#!/usr/bin/env python3
"""
Migration script to update packages for semantic chunking support.
Run this script to install the necessary dependencies.
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages for semantic chunking"""
    packages = [
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0"
    ]
    
    print("🔄 Installing packages for semantic chunking support...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("🎉 All packages installed successfully!")
    return True

def verify_installation():
    """Verify that the packages are properly installed"""
    try:
        from vector_store import CustomSemanticChunker
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        print("✅ CustomSemanticChunker import successful")
        print("✅ Scikit-learn cosine_similarity import successful")
        print("✅ NumPy import successful")
        return True
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Semantic Chunking Migration Script")
    print("=" * 40)
    
    if install_packages():
        print("\n🔍 Verifying installation...")
        if verify_installation():
            print("\n🎉 Migration completed successfully!")
            print("You can now use semantic chunking in your research agent.")
        else:
            print("\n⚠️ Installation completed but verification failed.")
            print("Please check your environment and try again.")
    else:
        print("\n❌ Migration failed!")
        print("Please check your environment and try again.")
