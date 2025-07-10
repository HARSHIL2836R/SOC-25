"""
UI components and styling for the Research Paper Analysis Agent.
"""

import streamlit as st
from typing import Any, Dict, List

def apply_custom_css() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1e3a8a;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 2px solid transparent;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background: linear-gradient(135deg, #2563eb, #1e40af);
            color: #ffffff;
            border-left: 5px solid #60a5fa;
            margin-left: 2rem;
        }
        .user-message strong {
            color: #bfdbfe;
        }
        .bot-message {
            background: linear-gradient(135deg, #1f2937, #374151);
            color: #f9fafb;
            border-left: 5px solid #10b981;
            margin-right: 2rem;
        }
        .bot-message strong {
            color: #6ee7b7;
        }
        .sidebar-info {
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid #cbd5e1;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

def display_header() -> None:
    """Display the main application header."""
    st.markdown('<h1 class="main-header">📄 Research Paper Analysis Agent</h1>', unsafe_allow_html=True)

def display_chat_message(role: str, content: str) -> None:
    """Display a chat message with appropriate styling."""
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message bot-message"><strong>Agent:</strong> {content}</div>', 
            unsafe_allow_html=True
        )

def display_paper_info(metadata: Dict[str, Any]) -> None:
    """Display paper information in the sidebar."""
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 📋 Paper Information")
    st.write(f"**Title:** {metadata.get('title', 'N/A')}")
    st.write(f"**Authors:** {metadata.get('authors', 'N/A')}")
    st.write(f"**Pages:** {metadata.get('total_pages', 'N/A')}")
    if 'local_path' in metadata:
        st.write(f"**File:** {metadata['local_path']}")
    st.markdown('</div>', unsafe_allow_html=True)

def display_how_to_use() -> None:
    """Display the how-to-use information."""
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Upload a PDF** or **enter a URL** in the sidebar
    2. **Wait for processing** - the agent will extract and index the content
    3. **Start chatting** with your paper using natural language
    4. **Use quick actions** for common queries
    5. **Explore sections** to understand the paper structure
    
    **Supported URLs:**
    - arXiv papers (abstract or PDF links)
    - Direct PDF URLs
    - Research paper websites
    """)

def display_sample_papers(sample_papers: List[Any]) -> None:
    """Display sample papers for testing."""
    st.markdown("### 📚 Try These Sample Papers:")
    for title, url in sample_papers:
        if st.button(f"📄 {title}", key=url):
            st.session_state.sample_url = url
            st.info(f"Copy this URL to the sidebar: {url}")

def display_paper_sections(sections: Dict[str, str]) -> None:
    """Display paper sections in an expandable format."""
    st.header("📑 Paper Sections")
    if sections:
        for section_name, content in sections.items():
            with st.expander(f"📖 {section_name.title()}", expanded=False):
                # Show first 500 characters
                preview = content[:500] + "..." if len(content) > 500 else content
                st.text_area(
                    f"Content ({len(content)} chars)",
                    preview,
                    height=200,
                    key=f"section_{section_name}"
                )
    else:
        st.info("No sections extracted from the paper.")

def display_search_indicator(status: str, query: str = "") -> None:
    """Display search status indicator"""
    if status == "searching":
        st.info(f"🔍 **Searching the internet for similar papers...**\n\n*Query: {query}*")
    elif status == "found":
        st.success("✅ **Similar papers found!** Results are being integrated into the response.")
    elif status == "no_results":
        st.warning("⚠️ **No similar papers found** in the search. Continuing with local analysis only.")
    elif status == "error":
        st.error("❌ **Search unavailable** - Using local paper analysis only.")
