# 📃 Research Paper Analysis Agent

An advanced AI-powered Streamlit application that revolutionizes how researchers interact with academic papers. Upload research papers and engage in intelligent conversations about their content, methodology, findings, and related work through a sophisticated LangGraph-powered agent.

## 🌟 Key Features

### 🔬 **Intelligent Paper Analysis**
- **Smart PDF Processing**: Upload research papers with automatic section extraction and metadata parsing
- **URL Support**: Directly analyze papers from arXiv URLs and other academic sources
- **Interactive Chat**: Natural language conversations about your research paper
- **Quick Actions**: One-click analysis for summaries, methodology, key findings, and more

### 🔍 **Internet-Enhanced Research**
- **Similar Papers Discovery**: Automatically find and analyze related research from academic databases
- **Comparative Analysis**: Get insights comparing your paper with recent similar research
- **Multi-Source Intelligence**: Combines local paper content with external research findings
- **Real-time Search**: Powered by advanced search APIs for up-to-date academic information

### 🧠 **Advanced AI Architecture**
- **LangGraph Agents**: Sophisticated agent workflow with intelligent routing and decision-making
- **Semantic Chunking**: Custom implementation for better content understanding and context preservation
- **Vector Search**: FAISS-powered similarity search for efficient content retrieval
- **Memory Management**: Maintains conversation context for multi-turn interactions

### 🎨 **User Experience**
- **Modern UI**: Clean, intuitive Streamlit interface with responsive design
- **Paper Management**: Save, load, and organize your research paper collection
- **Real-time Processing**: Live feedback and progress indicators
- **Quick Actions**: Pre-configured queries for common research tasks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for LLM and search services

### 1. Installation

```bash
# Navigate to the application directory
cd UI_agent

# Install required dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the `UI_agent` directory:

```env
# Required for LLM functionality
GROQ_API_KEY=your_groq_api_key_here

# Optional for internet search (highly recommended)
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get your API keys:**
- **GROQ API**: Sign up at [console.groq.com](https://console.groq.com/) for fast LLM inference
- **Tavily API**: Get your key at [tavily.com](https://tavily.com/) for academic search capabilities

### 3. Launch the Application

```bash
# Windows
start_app.bat

# Or manually
streamlit run app.py
```

### 4. Start Analyzing Papers!

1. **Upload a PDF** or **enter an arXiv URL** in the sidebar
2. **Wait for processing** (automatic section extraction and vectorization)
3. **Start chatting** with your paper or use quick action buttons
4. **Discover similar papers** with the enhanced search feature

## 🏗️ Project Architecture

### Modular Design
The application is built with a clean, modular architecture for maintainability and extensibility:

```
UI_agent/
├── app.py                 # Main Streamlit application entry point
├── config.py              # Configuration settings and constants
├── ui_components.py       # UI styling and Streamlit components
├── llm_utils.py          # LLM and embeddings initialization
├── pdf_processor.py      # PDF processing and metadata extraction
├── vector_store.py       # Vector store creation and semantic chunking
├── research_chain.py     # LangGraph agent setup and workflows
├── requirements.txt      # Python dependencies
├── start_app.bat         # Windows start script
└── papers/               # Directory for storing papers
```

### Core Modules

#### `config.py` - Configuration Management
- Application settings and UI constants
- Processing parameters (chunk sizes, similarity thresholds)
- API configurations and model settings
- Semantic chunking configuration

#### `ui_components.py` - User Interface
- Custom CSS styling for modern appearance
- Streamlit component wrappers
- Chat message formatting and display
- Paper information and section displays

#### `llm_utils.py` - AI Model Management
- LLM initialization with caching
- Embeddings model setup and management
- Resource optimization and error handling

#### `pdf_processor.py` - Document Processing
- Advanced PDF text extraction with section parsing
- Metadata extraction (title, authors, abstract)
- URL processing for arXiv and direct PDF links
- File management for uploaded/downloaded papers

#### `vector_store.py` - Knowledge Base
- **Custom Semantic Chunking**: Advanced text segmentation using sentence embeddings
- FAISS vector store creation and management
- Document metadata integration
- Configurable chunking strategies

#### `research_chain.py` - AI Agent Workflows
- LangGraph agent architecture
- Intelligent routing between local and internet search
- Multi-agent workflows for comprehensive analysis
- Context-aware response generation

## 🧠 Advanced Features

### Custom Semantic Chunking

Our proprietary semantic chunking implementation creates more coherent text segments compared to simple character-based splitting:

1. **Sentence-Level Analysis**: Text is intelligently split into semantic units
2. **Embedding-Based Similarity**: Uses sentence embeddings to measure content similarity
3. **Statistical Thresholds**: Configurable breakpoint detection using percentile, standard deviation, or IQR methods
4. **Context Preservation**: Buffer zones around breakpoints maintain contextual coherence

**Configuration Options:**
```python
# In config.py
SEMANTIC_CHUNKING_ENABLED = True
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE = "percentile"  # or "standard_deviation", "interquartile"
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT = 95.0
SEMANTIC_BUFFER_SIZE = 1
```

### LangGraph Agent System

The application uses LangGraph for sophisticated agent workflows:

- **Intelligent Routing**: Automatically determines when to search for similar papers
- **Multi-Agent Coordination**: Seamlessly coordinates between local analysis and internet research
- **Dynamic Tool Selection**: Chooses appropriate tools based on query context
- **State Management**: Maintains conversation context across interactions

### Internet Research Integration

Enhanced with real-time academic search capabilities:

- **Similar Paper Discovery**: Finds related research automatically
- **Comparative Analysis**: Provides insights comparing papers
- **Academic Database Search**: Searches arXiv, Google Scholar, and more
- **Citation Analysis**: Identifies key references and connections

## 🛠️ Technical Implementation

### Text Processing Pipeline
1. **Document Ingestion**: PDF upload or URL download
2. **Section Extraction**: Intelligent parsing of academic paper structure
3. **Metadata Extraction**: Title, authors, abstract, and key information
4. **Semantic Chunking**: Advanced text segmentation for better understanding
5. **Vectorization**: Transform text chunks into searchable embeddings
6. **Storage**: FAISS vector database for efficient similarity search

### AI Model Stack
- **LLM**: Groq-powered fast inference (Llama 3.1 8B Instant)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **Search**: Tavily API for internet research capabilities

### Performance Optimizations
- **Caching**: LLM and embeddings model caching for faster responses
- **Lazy Loading**: Components loaded on demand
- **Efficient Chunking**: Optimized chunk sizes and overlaps
- **Memory Management**: Conversation buffer with configurable limits

## 📱 Usage Guide

### Basic Workflow

1. **Paper Upload**
   - Upload a PDF file using the sidebar file uploader
   - Or enter an arXiv URL (e.g., `https://arxiv.org/abs/2301.xxxxx`)
   - Wait for automatic processing and section extraction

2. **Interactive Analysis**
   - Use the chat interface to ask questions about the paper
   - Try quick action buttons for common queries:
     - 📝 **Summarize Paper**: Get a comprehensive overview
     - 🔬 **Methodology**: Understand the research approach
     - 📊 **Key Findings**: Extract main results and conclusions
     - 🔍 **Similar Papers**: Find related research automatically

3. **Advanced Features**
   - Browse extracted paper sections in the right panel
   - Manage your paper collection in the sidebar
   - Clear chat history or switch between papers

### Example Queries

**Basic Analysis:**
- "What is the main contribution of this paper?"
- "Explain the methodology used in this research"
- "What are the limitations mentioned by the authors?"

**Comparative Research:**
- "Find similar papers to this research"
- "How does this work compare to recent advances in the field?"
- "What are the key differences between this and related work?"

**Deep Dive:**
- "What datasets were used in the experiments?"
- "Can you explain the mathematical formulation in section 3?"
- "What future work do the authors suggest?"

## 🔧 Configuration Options

### Chunking Configuration
Customize text processing in `config.py`:

```python
# Standard chunking
CHUNK_SIZE = 1000                    # Characters per chunk
CHUNK_OVERLAP = 200                  # Overlap between chunks

# Semantic chunking (recommended)
SEMANTIC_CHUNKING_ENABLED = True
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE = "percentile"
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT = 95.0
SEMANTIC_BUFFER_SIZE = 1
```

### Model Configuration
```python
# LLM settings
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000

# Embeddings
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_SEARCH_K = 6              # Number of relevant chunks to retrieve
```

## 📦 Dependencies

### Core Requirements
```
streamlit>=1.28.0          # Web application framework
pymupdf>=1.23.0           # PDF processing
python-dotenv>=1.0.0      # Environment variables
```

### AI/ML Stack
```
langgraph>=0.2.0          # Advanced agent workflows
langchain>=0.1.0          # LLM framework
langchain-groq>=0.1.0     # Groq LLM integration
langchain-huggingface     # Hugging Face embeddings
sentence-transformers     # Sentence embedding models
faiss-cpu>=1.7.4          # Vector similarity search
scikit-learn>=1.0.0       # Cosine similarity calculations
```

### Search Integration
```
tavily-python>=0.3.0      # Internet search API
```

## 🧩 Extending the Application

### Adding New Quick Actions
1. Define your action in the `render_quick_actions()` function in `app.py`
2. Create a handler function following the pattern of `handle_quick_action()`
3. Add appropriate prompts and processing logic

### Custom Chunking Strategies
1. Extend the `create_text_splitter()` function in `vector_store.py`
2. Implement your custom chunking logic
3. Update configuration options in `config.py`

### New Agent Capabilities
1. Modify the agent workflow in `research_chain.py`
2. Add new tools or modify existing ones
3. Update routing logic for different types of queries

## 🐛 Troubleshooting

### Common Issues

**"Failed to initialize LLM"**
- Check your GROQ_API_KEY in the `.env` file
- Verify your internet connection
- Ensure the API key is valid and has sufficient credits

**"Error processing PDF"**
- Ensure the PDF is text-readable (not just scanned images)
- Check file size (very large files may time out)
- Try downloading the PDF manually and uploading it

**"Internet search not available"**
- Add TAVILY_API_KEY to your `.env` file
- The similar papers feature requires this API key
- Basic paper analysis works without internet search

**Slow Processing**
- First-time model loading can be slow
- Subsequent runs are faster due to caching
- Consider upgrading to faster hardware for better performance

### Performance Tips
- Keep chunk sizes reasonable (1000-2000 characters)
- Use semantic chunking for better results
- Clear chat history periodically to manage memory
- Close other resource-intensive applications

## 🤝 Contributing

This project was developed as part of the IIT academics SOC@25 program. Future enhancements could include:

- **Multi-language Support**: Support for non-English research papers
- **Citation Network Analysis**: Visualize paper relationships and citations
- **Collaborative Features**: Share analyses and discussions with team members
- **Export Capabilities**: Save conversations and insights in various formats
- **Advanced Visualizations**: Interactive charts and graphs for paper insights

## 📚 Learning Resources

This project demonstrates advanced concepts in:

- **Large Language Models (LLMs)**: Practical application of modern AI
- **Vector Databases**: Efficient similarity search and retrieval
- **Semantic Processing**: Understanding text meaning beyond keywords
- **Agent Architectures**: Building intelligent, autonomous AI systems
- **Web Application Development**: Modern Streamlit development practices

For more learning resources, visit the [Notion page](https://responsible-minibus-af3.notion.site/LLM-Powered-Research-Agent-SOC-1fa6bc81b71780fab5f8d992834af017).

## 📄 License

This project is developed for educational purposes as part of the IIT academics program.

---

**Built with ❤️ for researchers, by researchers**

*Empowering academic research through intelligent AI assistance*
