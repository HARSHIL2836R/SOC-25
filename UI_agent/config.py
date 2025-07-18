"""
Configuration settings and constants for the Research Paper Analysis Agent.
"""

import os
from dotenv import load_dotenv
from typing import Any, List

# Load environment variables
load_dotenv()

# Application settings
APP_TITLE: str = "Research Paper Analysis Agent"
APP_ICON: str = "📃"
LAYOUT: str = "wide"
SIDEBAR_STATE: str = "expanded"

# File and folder settings
PAPERS_FOLDER: str = "papers"
SUPPORTED_FILE_TYPES: List[str] = ["pdf"]

# Processing settings
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
SIMILARITY_SEARCH_K: int = 6

# Semantic chunking settings
SEMANTIC_CHUNKING_ENABLED: bool = True
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE: str = "percentile"  # "percentile", "standard_deviation", or "interquartile"
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: float = 95.0  # For percentile: 95th percentile
SEMANTIC_BUFFER_SIZE: int = 1  # Number of sentences to include on either side of a breakpoint

# LLM settings
DEFAULT_MODEL: str = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 1000

# Embeddings settings
EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
NORMALIZE_EMBEDDINGS: bool = True

# Text processing settings
TEXT_SPLITTER_SEPARATORS: List[str] = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
MIN_SECTION_LENGTH: int = 50
PREVIEW_LENGTH: int = 500

# URL patterns
ARXIV_ABSTRACT_PATTERN: str = 'arxiv.org/abs/'
ARXIV_PDF_BASE: str = "https://arxiv.org/pdf/{}.pdf"

# Internet search settings for LangGraph agent
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
MAX_SEARCH_RESULTS: int = 5
SEARCH_TIMEOUT: int = 30

# LangGraph agent settings
ENABLE_INTERNET_SEARCH: bool = True
SEARCH_DOMAINS: List[str] = ["arxiv.org", "scholar.google.com", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov"]

# Section patterns for paper parsing
SECTION_PATTERNS: List[str] = [
    r'^(abstract|introduction|related work|methodology|method|approach|implementation|results|discussion|conclusion|references|acknowledgments)',
    r'^\d+\.?\s+(abstract|introduction|related work|methodology|method|approach|implementation|results|discussion|conclusion|references|acknowledgments)',
]

# Sample papers for testing
SAMPLE_PAPERS: List[Any] = [
    ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
    ("BERT: Pre-training of Deep Bidirectional Transformers", "https://arxiv.org/abs/1810.04805"),
    ("GPT-3: Language Models are Few-Shot Learners", "https://arxiv.org/abs/2005.14165")
]

def ensure_papers_folder() -> str:
    """Ensure the papers folder exists."""
    if not os.path.exists(PAPERS_FOLDER):
        os.makedirs(PAPERS_FOLDER)
    return PAPERS_FOLDER
