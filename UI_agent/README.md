# Research Paper Analysis Agent

A sophisticated AI-powered application for analyzing research papers with intelligent conversation capabilities and internet research integration.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Run the application:**
   ```bash
   # Windows
   start_app.bat
   
   # Or manually
   streamlit run app.py
   ```

## Features

- 📄 **PDF Upload & Analysis**: Upload research papers and extract structured content
- 🔗 **URL Support**: Direct analysis from arXiv URLs and other academic sources  
- 🤖 **AI Chat Interface**: Natural language conversations about your research
- 🔍 **Similar Papers Search**: Find related research automatically via internet search
- 📊 **Quick Actions**: One-click analysis for summaries, methodology, and key findings
- 🧠 **Semantic Chunking**: Advanced text processing for better understanding

## Project Structure

```
├── app.py                 # Main Streamlit application
├── config.py              # Configuration and settings
├── ui_components.py       # UI components and styling
├── llm_utils.py          # LLM and embeddings management
├── pdf_processor.py      # PDF processing and metadata extraction
├── vector_store.py       # Vector store and semantic chunking
├── research_chain.py     # LangGraph agent workflows
├── requirements.txt      # Dependencies
├── start_app.bat         # Windows startup script
└── papers/               # Stored papers directory
```

For detailed documentation, features, and usage guide, see the [main project README](../readme.md).

## API Requirements

- **GROQ_API_KEY**: Required for LLM functionality ([Get yours here](https://console.groq.com/))
- **TAVILY_API_KEY**: Optional for internet search ([Get yours here](https://tavily.com/))

Built with ❤️ for researchers, by researchers.
