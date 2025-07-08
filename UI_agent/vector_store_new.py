"""
Vector store creation and management for the Research Paper Analysis Agent.
"""

import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TEXT_SPLITTER_SEPARATORS, MIN_SECTION_LENGTH,
    SEMANTIC_CHUNKING_ENABLED, SEMANTIC_BREAKPOINT_THRESHOLD_TYPE, 
    SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT, SEMANTIC_BUFFER_SIZE
)
from typing import Any, Dict, List, Tuple, Optional
import re


class CustomSemanticChunker:
    """Custom semantic chunker implementation using sentence embeddings"""
    
    def __init__(self, embeddings, breakpoint_threshold_type="percentile", 
                breakpoint_threshold_amount=95.0, buffer_size=1):
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = buffer_size
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex"""
        # Split on sentence endings, keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_similarities(self, sentences: List[str]) -> List[float]:
        """Calculate cosine similarities between consecutive sentences"""
        if len(sentences) < 2:
            return []
        
        # Get embeddings for all sentences
        embeddings = []
        for sentence in sentences:
            emb = self.embeddings.embed_query(sentence)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        return similarities
    
    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """Find breakpoints based on similarity threshold"""
        if not similarities:
            return []
        
        similarities = np.array(similarities)
        
        if self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(similarities, 100 - self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = np.mean(similarities) - self.breakpoint_threshold_amount * np.std(similarities)
        elif self.breakpoint_threshold_type == "interquartile":
            q1 = np.percentile(similarities, 25)
            q3 = np.percentile(similarities, 75)
            iqr = q3 - q1
            threshold = q1 - self.breakpoint_threshold_amount * iqr
        else:
            threshold = np.mean(similarities)
        
        # Find indices where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # +1 because similarity[i] is between sentence[i] and sentence[i+1]
        
        return breakpoints
    
    def _create_chunks_with_buffer(self, sentences: List[str], breakpoints: List[int]) -> List[str]:
        """Create chunks with buffer sentences around breakpoints"""
        if not breakpoints:
            return [' '.join(sentences)]
        
        chunks = []
        start = 0
        
        for breakpoint in breakpoints:
            # Add buffer before breakpoint
            end = min(breakpoint + self.buffer_size, len(sentences))
            chunk = ' '.join(sentences[start:end])
            if chunk.strip():
                chunks.append(chunk)
            
            # Start next chunk with buffer before breakpoint
            start = max(0, breakpoint - self.buffer_size)
        
        # Add final chunk
        if start < len(sentences):
            chunk = ' '.join(sentences[start:])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_documents(self, texts: List[str]) -> List[Document]:
        """Create documents with semantic chunking"""
        all_documents = []
        
        for text in texts:
            try:
                # Split into sentences
                sentences = self._split_into_sentences(text)
                
                if len(sentences) <= 1:
                    # If only one sentence, return as is
                    all_documents.append(Document(page_content=text))
                    continue
                
                # Calculate similarities
                similarities = self._calculate_similarities(sentences)
                
                # Find breakpoints
                breakpoints = self._find_breakpoints(similarities)
                
                # Create chunks
                chunks = self._create_chunks_with_buffer(sentences, breakpoints)
                
                # Convert to documents
                for chunk in chunks:
                    if len(chunk.strip()) >= MIN_SECTION_LENGTH:
                        all_documents.append(Document(page_content=chunk))
                        
            except Exception as e:
                st.warning(f"Error in semantic chunking, using original text: {e}")
                all_documents.append(Document(page_content=text))
        
        return all_documents

    def split_text(self, text: str) -> List[str]:
        """Split text method for compatibility with RecursiveCharacterTextSplitter interface"""
        docs = self.create_documents([text])
        return [doc.page_content for doc in docs]


def create_text_splitter(embeddings: Any) -> Any:
    """Create appropriate text splitter based on configuration"""
    try:
        if SEMANTIC_CHUNKING_ENABLED:
            st.info("🧠 Using custom semantic chunking for better context-aware text splitting")
            return CustomSemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=SEMANTIC_BREAKPOINT_THRESHOLD_TYPE,
                breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT,
                buffer_size=SEMANTIC_BUFFER_SIZE
            )
        else:
            st.info("📝 Using recursive character text splitter")
            return RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=TEXT_SPLITTER_SEPARATORS
            )
    except Exception as e:
        st.warning(f"Failed to create semantic chunker, falling back to recursive splitter: {e}")
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=TEXT_SPLITTER_SEPARATORS
        )


def create_vector_store(sections: Dict[str, str], metadata: Dict[str, Any], embeddings: Any) -> Tuple[Optional[Any], List[Any]]:
    """Create vector store from paper sections using semantic or recursive chunking"""
    try:
        # Create appropriate text splitter
        text_splitter = create_text_splitter(embeddings)
        chunking_method = "semantic" if SEMANTIC_CHUNKING_ENABLED else "recursive"
        
        documents: List[Document] = []
        
        for section_name, section_content in sections.items():
            if len(section_content.strip()) < MIN_SECTION_LENGTH:
                continue
            
            # Create chunks using the selected text splitter
            try:
                if SEMANTIC_CHUNKING_ENABLED:
                    # Custom semantic chunker returns Document objects directly
                    section_docs = text_splitter.create_documents([section_content])
                    
                    for i, doc in enumerate(section_docs):
                        # Update metadata for semantic chunks
                        doc.metadata.update({
                            "source": metadata.get('filename', 'unknown'),
                            "paper_title": metadata.get('title', 'Unknown Title'),
                            "authors": metadata.get('authors', 'Unknown Authors'),
                            "section": section_name,
                            "chunk_id": i,
                            "chunk_size": len(doc.page_content),
                            "chunking_method": chunking_method
                        })
                        documents.append(doc)
                else:
                    # Recursive text splitter returns text chunks
                    chunks = text_splitter.split_text(section_content)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": metadata.get('filename', 'unknown'),
                                "paper_title": metadata.get('title', 'Unknown Title'),
                                "authors": metadata.get('authors', 'Unknown Authors'),
                                "section": section_name,
                                "chunk_id": i,
                                "chunk_size": len(chunk),
                                "chunking_method": chunking_method
                            }
                        )
                        documents.append(doc)
            except Exception as chunk_error:
                st.warning(f"Error processing section '{section_name}': {chunk_error}")
                continue
        
        if documents:
            st.success(f"✅ Created {len(documents)} chunks using {chunking_method} chunking")
            vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
            return vector_store, documents
        else:
            st.warning("⚠️ No documents were created from the sections")
            return None, []
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        return None, []


def display_chunking_config() -> None:
    """Display current chunking configuration"""
    with st.expander("🔧 Chunking Configuration", expanded=False):
        if SEMANTIC_CHUNKING_ENABLED:
            st.info("**Custom Semantic Chunking** is enabled")
            st.write(f"- **Breakpoint Type**: {SEMANTIC_BREAKPOINT_THRESHOLD_TYPE}")
            st.write(f"- **Threshold Amount**: {SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT}")
            st.write(f"- **Buffer Size**: {SEMANTIC_BUFFER_SIZE}")
            st.write("✨ This method creates chunks based on semantic similarity, resulting in more coherent text segments.")
        else:
            st.info("**Recursive Character Text Splitter** is enabled")
            st.write(f"- **Chunk Size**: {CHUNK_SIZE}")
            st.write(f"- **Chunk Overlap**: {CHUNK_OVERLAP}")
            st.write(f"- **Separators**: {TEXT_SPLITTER_SEPARATORS[:3]}...")
            st.write("📝 This method creates fixed-size chunks with character-based splitting.")
