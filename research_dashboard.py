import streamlit as st
import pymupdf
import os
import urllib.request
from urllib.parse import urlparse
import re

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Research Paper Analysis Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Initialize session state
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

@st.cache_resource
def initialize_llm_and_embeddings():
    """Initialize LLM and embeddings (cached)"""
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1000
        )
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing LLM/Embeddings: {e}")
        return None, None

def extract_paper_sections(pdf_path):
    """Extract sections from a research paper PDF"""
    try:
        doc = pymupdf.open(pdf_path)
        
        section_patterns = [
            r'^(abstract|introduction|related work|methodology|method|approach|implementation|results|discussion|conclusion|references|acknowledgments)',
            r'^\d+\.?\s+(abstract|introduction|related work|methodology|method|approach|implementation|results|discussion|conclusion|references|acknowledgments)',
            r'^\d+\.\d+\.?\s+.*',
        ]
        
        sections = {}
        current_section = "content"
        section_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                is_section_header = False
                for pattern in section_patterns:
                    if re.match(pattern, line.lower()):
                        if section_content:
                            sections[current_section] = '\n'.join(section_content)
                        current_section = line.lower()
                        section_content = []
                        is_section_header = True
                        break
                
                if not is_section_header:
                    section_content.append(line)
        
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        doc.close()
        return sections
    except Exception as e:
        st.error(f"Error extracting paper sections: {e}")
        return {}

def extract_paper_metadata(pdf_path):
    """Extract basic metadata from research paper"""
    try:
        doc = pymupdf.open(pdf_path)
        
        first_page = doc[0].get_text()
        lines = first_page.split('\n')
        
        title = "Unknown Title"
        authors = "Unknown Authors"
        
        substantial_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        if substantial_lines:
            title = substantial_lines[0]
            if len(substantial_lines) > 1:
                for line in substantial_lines[1:4]:
                    if any(indicator in line.lower() for indicator in ['@', 'university', 'institute', 'college']):
                        authors = line
                        break
        
        metadata = {
            'title': title,
            'authors': authors,
            'filename': os.path.basename(pdf_path),
            'total_pages': doc.page_count
        }
        
        doc.close()
        return metadata
    except Exception as e:
        st.error(f"Error extracting paper metadata: {e}")
        return {}

def process_arxiv_url(arxiv_url):
    """Convert arXiv abstract URL to PDF URL"""
    if 'arxiv.org/abs/' in arxiv_url:
        paper_id = arxiv_url.split('/abs/')[-1]
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        return pdf_url
    return arxiv_url

def extract_paper_from_url(url):
    """Extract paper content from URL and save to papers folder"""
    try:
        if 'arxiv.org' in url:
            url = process_arxiv_url(url)
        
        # Create papers folder if it doesn't exist
        papers_folder = "papers"
        if not os.path.exists(papers_folder):
            os.makedirs(papers_folder)
        
        # Generate filename from URL
        if 'arxiv.org' in url:
            # Extract arXiv ID for filename
            paper_id = url.split('/')[-1].replace('.pdf', '')
            filename = f"arxiv_{paper_id}.pdf"
        else:
            # Use URL-based filename
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename.endswith('.pdf'):
                filename = filename + '.pdf' if filename else 'downloaded_paper.pdf'
        
        # Full path for the saved file
        file_path = os.path.join(papers_folder, filename)
        
        # Download and save the file
        st.info(f"📥 Downloading: {filename}")
        urllib.request.urlretrieve(url, file_path)
        st.success(f"✅ Saved to: {file_path}")
        
        # Extract content from saved file
        sections = extract_paper_sections(file_path)
        metadata = extract_paper_metadata(file_path)
        metadata['source_url'] = url
        metadata['local_path'] = file_path
        
        return sections, metadata
    except Exception as e:
        st.error(f"Error extracting paper from URL: {e}")
        return {}, {}

def create_vector_store(sections, metadata, embeddings):
    """Create vector store from paper sections"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        documents = []
        
        for section_name, section_content in sections.items():
            if len(section_content.strip()) < 50:
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

def setup_research_chain(vector_store, llm, metadata):
    """Setup the research chain"""
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        research_prompt = PromptTemplate(
            template="""You are an expert research assistant specializing in academic paper analysis.

PAPER INFORMATION:
Title: {paper_title}
Authors: {authors}

INSTRUCTIONS:
- Analyze the provided research paper content to answer questions accurately
- Reference specific sections, methodologies, results, and findings when relevant
- Maintain academic rigor and cite evidence from the paper
- Use the conversation history to provide coherent, contextual responses
- If the question requires information not in the provided context, clearly state the limitations
- For technical questions, explain concepts clearly while maintaining accuracy

CONVERSATION HISTORY:
{chat_history}

RELEVANT PAPER CONTENT:
{context}

RESEARCH QUESTION: {question}

ANSWER:""",
            input_variables=['context', 'question', 'chat_history', 'paper_title', 'authors']
        )
        
        def format_research_docs(retrieved_docs):
            formatted_content = []
            for doc in retrieved_docs:
                section = doc.metadata.get('section', 'Unknown Section')
                content = doc.page_content
                formatted_content.append(f"[Section: {section}]\n{content}")
            return "\n\n" + "="*50 + "\n\n".join(formatted_content)
        
        research_parallel_chain = RunnableParallel({
            'context': lambda inputs: format_research_docs(retriever.invoke(inputs.get('question', ''))),
            'question': lambda inputs: inputs.get('question', ''),
            'chat_history': lambda inputs: inputs.get('chat_history', ''),
            'paper_title': lambda inputs: metadata.get('title', 'Unknown Title'),
            'authors': lambda inputs: metadata.get('authors', 'Unknown Authors')
        })
        
        parser = StrOutputParser()
        research_chain = research_parallel_chain | research_prompt | llm | parser
        
        return research_chain
    except Exception as e:
        st.error(f"Error setting up research chain: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Research Paper Analysis Agent</h1>', unsafe_allow_html=True)
    
    # Initialize LLM and embeddings
    llm, embeddings = initialize_llm_and_embeddings()
    
    if llm is None or embeddings is None:
        st.error("Failed to initialize LLM or embeddings. Please check your API keys and internet connection.")
        return
    
    # Sidebar for paper upload/URL input
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
                    # Create papers folder if it doesn't exist
                    papers_folder = "papers"
                    if not os.path.exists(papers_folder):
                        os.makedirs(papers_folder)
                    
                    # Save uploaded file to papers folder
                    file_path = os.path.join(papers_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ Saved to: {file_path}")
                    
                    try:
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
                                st.session_state.research_chain = setup_research_chain(vector_store, llm, metadata)
                                
                                st.success("✅ Paper processed successfully!")
                            else:
                                st.error("Failed to create vector store from the paper.")
                        else:
                            st.error("Failed to extract content from the PDF.")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                        # Clean up file if processing failed
                        if os.path.exists(file_path):
                            os.unlink(file_path)
        
        else:  # Enter URL
            url_input = st.text_input(
                "Enter paper URL",
                placeholder="https://arxiv.org/abs/2301.xxxxx or direct PDF URL",
                help="Enter arXiv URL or direct PDF link"
            )
            
            if url_input and st.button("Process URL"):
                with st.spinner("Processing URL..."):
                    sections, metadata = extract_paper_from_url(url_input)
                    
                    if sections:
                        # Create vector store
                        vector_store, documents = create_vector_store(sections, metadata, embeddings)
                        
                        if vector_store:
                            # Update session state
                            st.session_state.paper_loaded = True
                            st.session_state.paper_sections = sections
                            st.session_state.paper_metadata = metadata
                            st.session_state.vector_store = vector_store
                            st.session_state.research_chain = setup_research_chain(vector_store, llm, metadata)
                            
                            st.success("✅ Paper processed successfully!")
                        else:
                            st.error("Failed to create vector store from the paper.")
                    else:
                        st.error("Failed to extract content from the URL.")
        
        # Display paper info if loaded
        if st.session_state.paper_loaded:
            st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
            st.markdown("### 📋 Paper Information")
            metadata = st.session_state.paper_metadata
            st.write(f"**Title:** {metadata.get('title', 'N/A')}")
            st.write(f"**Authors:** {metadata.get('authors', 'N/A')}")
            st.write(f"**Pages:** {metadata.get('total_pages', 'N/A')}")
            if 'local_path' in metadata:
                st.write(f"**File:** {metadata['local_path']}")
            
            if st.button("🗑️ Clear Paper"):
                st.session_state.paper_loaded = False
                st.session_state.paper_sections = {}
                st.session_state.paper_metadata = {}
                st.session_state.vector_store = None
                st.session_state.chat_history = []
                st.session_state.research_memory.clear()
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show saved papers section
        st.markdown("---")
        st.markdown("### 📚 Saved Papers")
        papers_folder = "papers"
        if os.path.exists(papers_folder):
            pdf_files = [f for f in os.listdir(papers_folder) if f.endswith('.pdf')]
            if pdf_files:
                selected_file = st.selectbox(
                    "Load a saved paper:",
                    ["Select a paper..."] + pdf_files,
                    key="saved_papers_selector"
                )
                
                if selected_file != "Select a paper..." and st.button("📂 Load Selected Paper"):
                    file_path = os.path.join(papers_folder, selected_file)
                    with st.spinner(f"Loading {selected_file}..."):
                        try:
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
                                    st.session_state.research_chain = setup_research_chain(vector_store, llm, metadata)
                                    st.session_state.chat_history = []  # Clear previous chat
                                    st.session_state.research_memory.clear()
                                    
                                    st.success(f"✅ Loaded {selected_file} successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to create vector store from the paper.")
                            else:
                                st.error("Failed to extract content from the PDF.")
                        except Exception as e:
                            st.error(f"Error loading paper: {e}")
                
                # Show paper management options
                if pdf_files:
                    st.write(f"📄 **{len(pdf_files)}** papers saved")
                    with st.expander("🗂️ Manage Papers", expanded=False):
                        for pdf_file in pdf_files:
                            file_path = os.path.join(papers_folder, pdf_file)
                            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📄 {pdf_file} ({file_size:.1f} MB)")
                            with col2:
                                if st.button("🗑️", key=f"delete_{pdf_file}", help=f"Delete {pdf_file}"):
                                    os.unlink(file_path)
                                    st.success(f"Deleted {pdf_file}")
                                    st.rerun()
            else:
                st.info("No saved papers yet. Upload or download papers to see them here.")
        else:
            st.info("Papers folder will be created when you upload or download your first paper.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.paper_loaded:
            st.header("💬 Chat with Your Paper")
            
            # Chat interface
            chat_container = st.container()
            
            # Display chat history
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message bot-message"><strong>Agent:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Chat input
            user_question = st.chat_input("Ask a question about the paper...")
            
            if user_question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                try:
                    # Get chat history from memory
                    chat_history = st.session_state.research_memory.load_memory_variables({})["chat_history"]
                    if isinstance(chat_history, list):
                        chat_history = "\n".join(str(x) for x in chat_history)
                    elif chat_history is None:
                        chat_history = ""
                    
                    # Get response from research chain
                    with st.spinner("Thinking..."):
                        result = st.session_state.research_chain.invoke({
                            "question": user_question,
                            "chat_history": chat_history
                        })
                        
                        # Extract answer
                        if isinstance(result, dict) and "answer" in result:
                            answer = result["answer"]
                        else:
                            answer = result
                        
                        # Save to memory
                        st.session_state.research_memory.save_context(
                            {"question": user_question}, 
                            {"answer": answer}
                        )
                        
                        # Add bot response to history
                        st.session_state.chat_history.append({"role": "bot", "content": answer})
                        
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing question: {e}")
            
            # Quick action buttons
            st.markdown("### 🚀 Quick Actions")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("📝 Summarize Paper"):
                    quick_question = "Please provide a comprehensive summary of this research paper."
                    st.session_state.chat_history.append({"role": "user", "content": quick_question})
                    
                    try:
                        # Get chat history from memory
                        chat_history = st.session_state.research_memory.load_memory_variables({})["chat_history"]
                        if isinstance(chat_history, list):
                            chat_history = "\n".join(str(x) for x in chat_history)
                        elif chat_history is None:
                            chat_history = ""
                        
                        # Get response from research chain
                        with st.spinner("Generating summary..."):
                            result = st.session_state.research_chain.invoke({
                                "question": quick_question,
                                "chat_history": chat_history
                            })
                            
                            # Extract answer
                            if isinstance(result, dict) and "answer" in result:
                                answer = result["answer"]
                            else:
                                answer = result
                            
                            # Save to memory
                            st.session_state.research_memory.save_context(
                                {"question": quick_question}, 
                                {"answer": answer}
                            )
                            
                            # Add bot response to history
                            st.session_state.chat_history.append({"role": "bot", "content": answer})
                            
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            
            with col_b:
                if st.button("🔬 Methodology"):
                    quick_question = "What methodology does this paper use? Explain the experimental setup."
                    st.session_state.chat_history.append({"role": "user", "content": quick_question})
                    
                    try:
                        # Get chat history from memory
                        chat_history = st.session_state.research_memory.load_memory_variables({})["chat_history"]
                        if isinstance(chat_history, list):
                            chat_history = "\n".join(str(x) for x in chat_history)
                        elif chat_history is None:
                            chat_history = ""
                        
                        # Get response from research chain
                        with st.spinner("Analyzing methodology..."):
                            result = st.session_state.research_chain.invoke({
                                "question": quick_question,
                                "chat_history": chat_history
                            })
                            
                            # Extract answer
                            if isinstance(result, dict) and "answer" in result:
                                answer = result["answer"]
                            else:
                                answer = result
                            
                            # Save to memory
                            st.session_state.research_memory.save_context(
                                {"question": quick_question}, 
                                {"answer": answer}
                            )
                            
                            # Add bot response to history
                            st.session_state.chat_history.append({"role": "bot", "content": answer})
                            
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error analyzing methodology: {e}")
            
            with col_c:
                if st.button("📊 Key Findings"):
                    quick_question = "What are the key findings and results of this paper?"
                    st.session_state.chat_history.append({"role": "user", "content": quick_question})
                    
                    try:
                        # Get chat history from memory
                        chat_history = st.session_state.research_memory.load_memory_variables({})["chat_history"]
                        if isinstance(chat_history, list):
                            chat_history = "\n".join(str(x) for x in chat_history)
                        elif chat_history is None:
                            chat_history = ""
                        
                        # Get response from research chain
                        with st.spinner("Finding key results..."):
                            result = st.session_state.research_chain.invoke({
                                "question": quick_question,
                                "chat_history": chat_history
                            })
                            
                            # Extract answer
                            if isinstance(result, dict) and "answer" in result:
                                answer = result["answer"]
                            else:
                                answer = result
                            
                            # Save to memory
                            st.session_state.research_memory.save_context(
                                {"question": quick_question}, 
                                {"answer": answer}
                            )
                            
                            # Add bot response to history
                            st.session_state.chat_history.append({"role": "bot", "content": answer})
                            
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error finding key findings: {e}")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.research_memory.clear()
                st.rerun()
        
        else:
            st.info("👈 Please upload a PDF or enter a URL in the sidebar to start analyzing a research paper.")
            
            # Sample URLs for testing
            st.markdown("### 📚 Try These Sample Papers:")
            sample_papers = [
                ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
                ("BERT: Pre-training of Deep Bidirectional Transformers", "https://arxiv.org/abs/1810.04805"),
                ("GPT-3: Language Models are Few-Shot Learners", "https://arxiv.org/abs/2005.14165")
            ]
            
            for title, url in sample_papers:
                if st.button(f"📄 {title}", key=url):
                    st.session_state.sample_url = url
                    st.info(f"Copy this URL to the sidebar: {url}")
    
    with col2:
        if st.session_state.paper_loaded:
            st.header("📑 Paper Sections")
            sections = st.session_state.paper_sections
            
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
        
        else:
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

if __name__ == "__main__":
    main()
