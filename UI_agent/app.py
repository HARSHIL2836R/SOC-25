"""
Main Streamlit application for the Research Paper Analysis Agent.
"""

import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from typing import Any, Dict, List, Optional, Tuple, Union

# Import custom modules
from config import (
    APP_TITLE, APP_ICON, LAYOUT, SIDEBAR_STATE, PAPERS_FOLDER, SAMPLE_PAPERS
)
from ui_components import (
    apply_custom_css, display_header, display_chat_message, display_paper_info,
    display_how_to_use, display_sample_papers, display_paper_sections
)
from llm_utils import initialize_llm_and_embeddings
from pdf_processor import (
    extract_paper_sections, extract_paper_metadata, extract_paper_from_url,
    save_uploaded_file, get_saved_papers, delete_paper
)
from vector_store import create_vector_store, display_chunking_config
from research_chain import setup_research_chain, process_question, display_search_status

def initialize_session_state() -> None:
    """Initialize session state variables"""
    if 'paper_loaded' not in st.session_state:
        st.session_state.paper_loaded = False
    if 'paper_metadata' not in st.session_state:
        st.session_state.paper_metadata = {}
    if 'paper_sections' not in st.session_state:
        st.session_state.paper_sections = {}
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'research_memory' not in st.session_state:
        st.session_state.research_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=False
        )
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def process_uploaded_pdf(uploaded_file: Any, embeddings: Any, llm: Any) -> bool:
    """Process an uploaded PDF file"""
    try:
        file_path = save_uploaded_file(uploaded_file)
        if not file_path:
            return False
        
        # Extract sections and metadata
        sections = extract_paper_sections(file_path)
        metadata = extract_paper_metadata(file_path)
        metadata['local_path'] = file_path
        
        if sections:
            # Create vector store
            vector_store, documents = create_vector_store(sections, metadata, embeddings)
            
            if vector_store:
                # Update session state
                st.session_state.paper_loaded = True
                st.session_state.paper_sections = sections
                st.session_state.paper_metadata = metadata
                st.session_state.vector_store = vector_store
                st.session_state.research_agent = setup_research_chain(vector_store, llm, metadata)
                
                st.success("✅ Paper processed successfully!")
                return True
            else:
                st.error("Failed to create vector store from the paper.")
        else:
            st.error("Failed to extract content from the PDF.")
        
        # Clean up file if processing failed
        if os.path.exists(file_path):
            os.unlink(file_path)
        return False
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def process_url_input(url: str, embeddings: Any, llm: Any) -> bool:
    """Process a URL input"""
    try:
        sections, metadata = extract_paper_from_url(url)
        
        if sections:
            # Create vector store
            vector_store, documents = create_vector_store(sections, metadata, embeddings)
            
            if vector_store:
                # Update session state
                st.session_state.paper_loaded = True
                st.session_state.paper_sections = sections
                st.session_state.paper_metadata = metadata
                st.session_state.vector_store = vector_store
                st.session_state.research_agent = setup_research_chain(vector_store, llm, metadata)
                
                st.success("✅ Paper processed successfully!")
                return True
            else:
                st.error("Failed to create vector store from the paper.")
        else:
            st.error("Failed to extract content from the URL.")
        return False
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return False

def load_saved_paper(filename: str, embeddings: Any, llm: Any) -> bool:
    """Load a saved paper"""
    try:
        file_path = os.path.join(PAPERS_FOLDER, filename)
        sections = extract_paper_sections(file_path)
        metadata = extract_paper_metadata(file_path)
        metadata['local_path'] = file_path
        
        if sections:
            vector_store, documents = create_vector_store(sections, metadata, embeddings)
            
            if vector_store:
                st.session_state.paper_loaded = True
                st.session_state.paper_sections = sections
                st.session_state.paper_metadata = metadata
                st.session_state.vector_store = vector_store
                st.session_state.research_agent = setup_research_chain(vector_store, llm, metadata)
                st.session_state.chat_history = []  # Clear previous chat
                st.session_state.research_memory.clear()
                
                st.success(f"✅ Loaded {filename} successfully!")
                return True
            else:
                st.error("Failed to create vector store from the paper.")
        else:
            st.error("Failed to extract content from the PDF.")
        return False
    except Exception as e:
        st.error(f"Error loading paper: {e}")
        return False

def render_sidebar(llm: Any, embeddings: Any) -> None:
    """Render the sidebar with paper loading options"""
    with st.sidebar:
        st.header("📂 Load Research Paper")
        
        option = st.radio("Choose input method:", ["Upload PDF", "Enter URL"])
        
        if option == "Upload PDF":
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a research paper in PDF format"
            )
            
            if uploaded_file is not None and st.button("Process PDF"):
                with st.spinner("Processing uploaded PDF..."):
                    process_uploaded_pdf(uploaded_file, embeddings, llm)
        
        else:  # Enter URL
            url_input = st.text_input(
                "Enter paper URL",
                placeholder="https://arxiv.org/abs/2301.xxxxx or direct PDF URL",
                help="Enter arXiv URL or direct PDF link"
            )
            
            if url_input and st.button("Process URL"):
                with st.spinner("Processing URL..."):
                    process_url_input(url_input, embeddings, llm)
        
        # Display paper info if loaded
        if st.session_state.paper_loaded:
            display_paper_info(st.session_state.paper_metadata)
            
            if st.button("🗑️ Clear Paper"):
                st.session_state.paper_loaded = False
                st.session_state.paper_sections = {}
                st.session_state.paper_metadata = {}
                st.session_state.vector_store = None
                st.session_state.chat_history = []
                st.session_state.research_memory.clear()
                st.rerun()
        
        # Show saved papers section
        st.markdown("---")
        st.markdown("### 📚 Saved Papers")
        pdf_files = get_saved_papers()
        
        if pdf_files:
            selected_file = st.selectbox(
                "Load a saved paper:",
                ["Select a paper..."] + pdf_files,
                key="saved_papers_selector"
            )
            
            if selected_file != "Select a paper..." and st.button("📂 Load Selected Paper"):
                with st.spinner(f"Loading {selected_file}..."):
                    if load_saved_paper(selected_file, embeddings, llm):
                        st.rerun()
            
            # Show paper management options
            st.write(f"📄 **{len(pdf_files)}** papers saved")
            with st.expander("🗂️ Manage Papers", expanded=False):
                for pdf_file in pdf_files:
                    file_path = os.path.join(PAPERS_FOLDER, pdf_file)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📄 {pdf_file} ({file_size:.1f} MB)")
                        with col2:
                            if st.button("🗑️", key=f"delete_{pdf_file}", help=f"Delete {pdf_file}"):
                                if delete_paper(pdf_file):
                                    st.success(f"Deleted {pdf_file}")
                                    st.rerun()
        else:
            st.info("No saved papers yet. Upload or download papers to see them here.")
        
        # Display chunking configuration
        st.markdown("---")
        
        # Display internet search status
        display_search_status()
        display_chunking_config()

def handle_quick_action(question: str) -> None:
    """Handle quick action button press"""
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    try:
        # Special handling for similar papers search
        if "similar papers" in question.lower() or "find similar" in question.lower():
            with st.spinner("🔍 Searching for similar papers on the internet..."):
                answer = process_question(
                    st.session_state.research_agent,
                    st.session_state.research_memory,
                    question
                )
        else:
            with st.spinner("Processing..."):
                answer = process_question(
                    st.session_state.research_agent,
                    st.session_state.research_memory,
                    question
                )
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "bot", "content": answer})
        st.rerun()
    except Exception as e:
        st.error(f"Error processing question: {e}")

def handle_similar_papers_action() -> None:
    """Special handler for similar papers quick action with enhanced UI"""
    question = "Find similar papers to this research and provide a comparative analysis with key differences and similarities."
    
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": "🔍 Find Similar Papers"})
    
    try:
        # Show search progress
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info("🔍 **Searching for similar papers...**\n\nThis may take a moment as we search academic databases.")
        
        # Process the question
        answer = process_question(
            st.session_state.research_agent,
            st.session_state.research_memory,
            question
        )
        
        # Clear progress indicator and add response
        progress_placeholder.empty()
        st.session_state.chat_history.append({"role": "bot", "content": answer})
        st.rerun()
        
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Error searching for similar papers: {e}")

def render_chat_interface() -> None:
    """Render the chat interface"""
    st.header("💬 Chat with Your Paper")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask a question about the paper...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        try:
            with st.spinner("Thinking..."):
                answer = process_question(
                    st.session_state.research_agent,
                    st.session_state.research_memory,
                    user_question
                )
                
                # Add bot response to history
                st.session_state.chat_history.append({"role": "bot", "content": answer})
                st.rerun()
        except Exception as e:
            st.error(f"Error processing question: {e}")

def render_quick_actions() -> None:
    """Render quick action buttons"""
    st.markdown("### 🚀 Quick Actions")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        if st.button("📝 Summarize Paper"):
            handle_quick_action("Please provide a comprehensive summary of this research paper.")
    
    with col_b:
        if st.button("🔬 Methodology"):
            handle_quick_action("What methodology does this paper use? Explain the experimental setup.")
    
    with col_c:
        if st.button("📊 Key Findings"):
            handle_quick_action("What are the key findings and results of this paper?")
    
    with col_d:
        if st.button("🔍 Similar Papers", help="Search the internet for papers similar to this research"):
            handle_similar_papers_action()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.research_memory.clear()
        st.rerun()

def main() -> None:
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    
    # Apply custom CSS and display header
    apply_custom_css()
    display_header()
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize LLM and embeddings
    llm, embeddings = initialize_llm_and_embeddings()
    
    if llm is None or embeddings is None:
        st.error("Failed to initialize LLM or embeddings. Please check your API keys and internet connection.")
        return
    
    # Render sidebar
    render_sidebar(llm, embeddings)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.paper_loaded:
            render_chat_interface()
            render_quick_actions()
        else:
            st.info("👈 Please upload a PDF or enter a URL in the sidebar to start analyzing a research paper.")
            
            # # Highlight new feature
            # st.markdown("### 🆕 New Feature: Similar Papers Search")
            # st.success("""
            # 🔍 **Discover Related Research Automatically!**
            
            # Once you upload a paper, you can:
            # - Click the **"🔍 Similar Papers"** quick action button
            # - Ask questions like *"Find similar papers to this research"*
            # - Get comparative analysis with related work from academic databases
            
            # *Powered by internet search across arXiv, Google Scholar, and more!*
            # """)
            
            display_sample_papers(SAMPLE_PAPERS)
    
    with col2:
        if st.session_state.paper_loaded:
            display_paper_sections(st.session_state.paper_sections)
        else:
            display_how_to_use()

if __name__ == "__main__":
    main()
