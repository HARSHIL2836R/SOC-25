# Research Paper Analysis Agent - LangGraph Enhanced

A powerful Streamlit application that allows you to upload research papers and interact with them using advanced AI agents powered by LangGraph. The agent can not only analyze your paper but also search the internet for similar research and provide comparative insights.

## 🆕 What's New - LangGraph Integration

- **Enhanced Agent Architecture**: Migrated from LangChain to LangGraph for better control flow and agent behavior
- **Internet Search Capability**: Automatic search for similar papers when relevant queries are detected
- **Multi-Agent Workflow**: Intelligent routing between local paper analysis and internet research
- **Comparative Analysis**: Get insights comparing your paper with recent similar research

## 🔧 Features

### Core Functionality
- **PDF Upload & Processing**: Upload research papers and extract structured content
- **URL Support**: Direct analysis from arXiv URLs and other academic sources
- **Smart Chunking**: Advanced semantic chunking for better content understanding
- **Interactive Chat**: Natural language conversations about your research paper

### LangGraph Agent Features
- **Intelligent Routing**: Automatically determines when to search for similar papers
- **Internet Research**: Searches academic databases for related work
- **Source Integration**: Combines local paper content with external research findings
- **Contextual Responses**: Provides comprehensive answers using multiple information sources

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd UI_agent

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Setup

Create a `.env` file in the project root (copy from `.env.example`):

```env
# Required for LLM functionality
GROQ_API_KEY=your_groq_api_key_here

# Optional for internet search (highly recommended)
TAVILY_API_KEY=your_tavily_api_key_here
```

**API Key Sources:**
- **Groq API**: Get your free key at [console.groq.com](https://console.groq.com/keys)
- **Tavily API**: Get your key at [tavily.com](https://tavily.com/) (enables internet search)

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py

# Or use the batch file (Windows)
start_app.bat
```

## 🧠 How the LangGraph Agent Works

### Agent Workflow

1. **Question Analysis**: The agent analyzes your question to understand intent
2. **Routing Decision**: Determines whether internet search is needed based on keywords
3. **Information Gathering**: 
   - **Local Search**: Retrieves relevant content from your uploaded paper
   - **Internet Search**: Searches for similar papers when appropriate
4. **Response Generation**: Synthesizes information from all sources

### When Internet Search is Triggered

The agent automatically searches for similar papers when you ask about:
- "Similar papers" or "related work"
- "Recent research" or "latest developments"
- "Compare with other studies"
- "State of the art" or "current trends"
- "Benchmarks" or "surveys"

### Search Sources

The agent searches across:
- arXiv preprint server
- Google Scholar
- Semantic Scholar
- PubMed (for medical/biological research)
- IEEE Xplore
- ACM Digital Library

## 💡 Usage Examples

### Basic Paper Analysis
```
Q: "What is the main contribution of this paper?"
→ Analyzes your uploaded paper only
```

### Comparative Research
```
Q: "Are there similar papers to this research?"
→ Searches internet + analyzes your paper
→ Provides list of similar papers with summaries
```

### Technical Deep Dive
```
Q: "How does this methodology compare to recent approaches?"
→ Searches for recent papers with similar methodologies
→ Provides comparative analysis
```

## 🔧 Configuration

### Agent Settings (config.py)

```python
# Internet search settings
ENABLE_INTERNET_SEARCH = True
MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT = 30

# Search domains
SEARCH_DOMAINS = [
    "arxiv.org", 
    "scholar.google.com", 
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov"
]
```

### Model Settings

```python
# LLM configuration
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000
```

## 🛠️ Architecture

### LangGraph State Management

```python
class AgentState(TypedDict):
    messages: List[AnyMessage]
    question: str
    paper_title: str
    paper_authors: str
    chat_history: str
    context: str
    similar_papers: List[Dict[str, Any]]
    search_performed: bool
    current_task: str
```

### Agent Nodes

- **Router**: Analyzes questions and decides on workflow
- **Local Search**: Retrieves content from uploaded paper
- **Internet Search**: Finds similar papers online
- **Response Generator**: Synthesizes final answer

## 🔍 Troubleshooting

### Common Issues

1. **No Internet Search Results**
   - Check TAVILY_API_KEY in .env file
   - Verify internet connection
   - Try more specific search terms

2. **LLM Connection Issues**
   - Verify GROQ_API_KEY is correct
   - Check API quota/limits
   - Ensure stable internet connection

3. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Tips

- **For faster responses**: Disable internet search for basic questions
- **For better search results**: Use specific technical terms in your questions
- **For comprehensive analysis**: Ask about "similar work" or "related research"

## 📦 Dependencies

### Core LangGraph Stack
- `langgraph>=0.2.0` - Agent workflow management
- `langchain>=0.1.0` - LLM integration
- `langchain-groq>=0.1.0` - Groq LLM provider
- `tavily-python>=0.3.0` - Internet search capability

### Supporting Libraries
- `streamlit>=1.28.0` - Web interface
- `pymupdf>=1.23.0` - PDF processing
- `faiss-cpu>=1.7.4` - Vector search
- `sentence-transformers>=2.2.2` - Embeddings

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional search providers
- Enhanced paper parsing
- Better similarity detection
- Custom agent workflows
- Integration with reference managers

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
3. Open an issue on the project repository

---

**Note**: This application requires API keys to function. The Groq API key is required for basic functionality, while the Tavily API key enables the enhanced internet search features.
