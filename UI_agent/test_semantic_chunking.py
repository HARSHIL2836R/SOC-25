#!/usr/bin/env python3
"""
Test script to verify semantic chunking implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import CustomSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def test_semantic_chunking():
    """Test the custom semantic chunker"""
    print("🧪 Testing Custom Semantic Chunking")
    print("=" * 50)
    
    # Initialize embeddings
    print("📥 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embeddings loaded successfully")
    
    # Create semantic chunker
    chunker = CustomSemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
        buffer_size=1
    )
    
    # Test text with distinct topics
    test_text = """
    Artificial intelligence has revolutionized many fields in recent years. Machine learning algorithms can now process vast amounts of data with unprecedented accuracy. Deep learning models have shown remarkable performance in image recognition tasks.
    
    Climate change is one of the most pressing issues of our time. Rising global temperatures have led to melting ice caps and rising sea levels. Scientists are working on solutions to reduce carbon emissions and mitigate environmental damage.
    
    The field of quantum computing promises to solve problems that are intractable for classical computers. Quantum bits, or qubits, can exist in superposition states that allow for parallel computation. This technology could revolutionize cryptography and scientific simulations.
    """
    
    print(f"📝 Input text length: {len(test_text)} characters")
    
    # Test chunking
    try:
        documents = chunker.create_documents([test_text])
        print(f"✅ Created {len(documents)} semantic chunks")
        
        for i, doc in enumerate(documents, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Length: {len(doc.page_content)} characters")
            print(f"Content: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        return False
    
    print("\n🎉 Semantic chunking test completed successfully!")
    return True

if __name__ == "__main__":
    test_semantic_chunking()
