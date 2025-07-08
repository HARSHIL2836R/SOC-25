"""
Vector store creation and management for the Research Paper Analysis Agent.
"""

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TEXT_SPLITTER_SEPARATORS, MIN_SECTION_LENGTH
)
from typing import Any, Dict, List, Tuple, Optional

def create_vector_store(sections: Dict[str, str], metadata: Dict[str, Any], embeddings: Any) -> Tuple[Optional[Any], List[Any]]:
    """Create vector store from paper sections"""
    try:
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=TEXT_SPLITTER_SEPARATORS
        )
        
        documents: List[Document] = []
        
        for section_name, section_content in sections.items():
            if len(section_content.strip()) < MIN_SECTION_LENGTH:
                continue
                
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
                        "chunk_size": len(chunk)
                    }
                )
                documents.append(doc)
        
        if documents:
            vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
            return vector_store, documents
        else:
            return None, []
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, []
