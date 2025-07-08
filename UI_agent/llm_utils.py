"""
LLM and embeddings utilities for the Research Paper Analysis Agent.
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS,
    EMBEDDINGS_MODEL, NORMALIZE_EMBEDDINGS
)
from typing import Any, Tuple

@st.cache_resource
def initialize_llm_and_embeddings() -> Tuple[Any, Any]:
    """Initialize LLM and embeddings (cached)"""
    try:
        llm = ChatGroq(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
        )
        
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing LLM/Embeddings: {e}")
        return None, None
