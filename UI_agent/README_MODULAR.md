# Research Paper Analysis Agent - Modular Version

A modular Streamlit application for analyzing research papers using AI. This version has been refactored from a single monolithic file into organized, maintainable modules.

## 🏗️ Project Structure

```
├── app.py                  # Main Streamlit application entry point
├── config.py               # Configuration settings and constants
├── ui_components.py        # UI styling and Streamlit components
├── llm_utils.py           # LLM and embeddings initialization
├── pdf_processor.py       # PDF processing and metadata extraction
├── vector_store.py        # Vector store creation and management
├── research_chain.py      # Research chain setup and question processing
├── requirements.txt       # Python dependencies
├── start_app.bat          # Windows batch script to start the app
├── start_dashboard.bat    # Legacy start script (for old version)
├── research_dashboard.py  # Original monolithic file (kept for reference)
└── papers/                # Directory for storing uploaded/downloaded papers
```

## 📦 Modules Overview

### `config.py`
- Application configuration and constants
- Environment settings
- File paths and processing parameters
- Sample papers for testing

### `ui_components.py`
- Custom CSS styling
- Streamlit UI components
- Chat message formatting
- Paper information display functions

### `llm_utils.py`
- LLM (Language Model) initialization
- Embeddings model setup
- Cached resource management

### `pdf_processor.py`
- PDF text extraction and section parsing
- Metadata extraction from research papers
- URL processing for arXiv and direct PDF links
- File management for uploaded and downloaded papers

### `vector_store.py`
- Document chunking and vectorization
- FAISS vector store creation
- Document metadata management

### `research_chain.py`
- LangChain research pipeline setup
- Question processing and context retrieval
- Conversation memory management

### `app.py`
- Main Streamlit application
- Session state management
- UI orchestration and event handling

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for downloading models

### Installation & Setup

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application:**
   
   **Option A: Using the batch script (Windows)**
   ```bash
   start_app.bat
   ```
   
   **Option B: Using Streamlit directly**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for the LLM functionality

### Customization
You can modify settings in `config.py`:
- Change LLM model and parameters
- Adjust chunk size and overlap for document processing
- Modify UI settings and styling
- Add new sample papers

## 📝 Usage

1. **Load a Paper:**
   - Upload a PDF file, or
   - Enter an arXiv URL or direct PDF link

2. **Chat with the Paper:**
   - Ask questions about the content
   - Use quick action buttons for common queries
   - View paper sections in the sidebar

3. **Manage Papers:**
   - Save papers locally for future use
   - Delete unwanted papers
   - Switch between different papers

## 🔍 Features

- **Multi-format Support:** Upload PDFs or use URLs (arXiv, direct PDF links)
- **Intelligent Parsing:** Automatic section detection and metadata extraction
- **Semantic Search:** Vector-based content retrieval for accurate answers
- **Chat Interface:** Natural language conversation with your papers
- **Paper Management:** Local storage and organization of papers
- **Quick Actions:** Pre-defined queries for common research tasks

## 🛠️ Development

### Adding New Features

1. **New UI Components:** Add to `ui_components.py`
2. **Processing Logic:** Extend `pdf_processor.py` or `vector_store.py`
3. **Configuration:** Update `config.py` for new settings
4. **Main App Logic:** Modify `app.py` for new workflows

### Module Dependencies
```
app.py
├── config.py
├── ui_components.py
├── llm_utils.py
├── pdf_processor.py
├── vector_store.py
└── research_chain.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors:** Ensure all modules are in the same directory
2. **API Key Errors:** Check your `.env` file and GROQ_API_KEY
3. **PDF Processing Errors:** Verify the PDF is not corrupted or password-protected
4. **Memory Issues:** Try reducing chunk size in `config.py`

### Performance Tips

- Use the cached LLM initialization for faster startup
- Process smaller documents for quicker responses
- Clear chat history periodically to free memory

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in the appropriate module
4. Test the application thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the module documentation
3. Create an issue with detailed error information

---

**Note:** This modular version maintains all functionality of the original `research_dashboard.py` while providing better organization, maintainability, and extensibility.
