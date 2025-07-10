# Similar Papers Search Feature Demo

## Overview
The Research Paper Analysis Agent now includes a powerful "Similar Papers" search feature that automatically finds related research on the internet and provides comparative analysis.

## How It Works

### 1. Quick Action Button
- **Location**: In the Quick Actions section (🚀 Quick Actions)
- **Button**: 🔍 Similar Papers
- **Function**: Automatically searches for papers similar to your uploaded research

### 2. Automatic Keyword Detection
The agent automatically triggers internet search when you ask questions containing:
- "similar papers"
- "related work" 
- "recent research"
- "compare with other studies"
- "state of the art"
- "current trends"
- "benchmark studies"
- "survey papers"

### 3. Search Process
1. **Query Optimization**: Extracts key terms from your paper's title
2. **Internet Search**: Searches academic databases using Tavily API
3. **Result Filtering**: Filters for academic papers only
4. **Relevance Ranking**: Sorts results by relevance score
5. **Integration**: Combines findings with local paper analysis

## Example Usage

### Using Quick Action
1. Upload your research paper
2. Click "🔍 Similar Papers" button
3. Wait for search to complete
4. Review comparative analysis with similar papers

### Using Natural Language
```
User: "Are there any similar papers to this research?"
Agent: 🔍 Searching for similar papers...
       ✅ Found 5 similar papers
       [Provides detailed comparison]
```

```
User: "What are recent developments in this field?"
Agent: 🔍 Searching for recent research...
       [Returns latest papers with analysis]
```

## Search Sources
The agent searches across:
- **arXiv** - Preprint server
- **Google Scholar** - Academic search engine
- **Semantic Scholar** - AI-powered research tool
- **PubMed** - Medical/biological research
- **IEEE Xplore** - Engineering and computer science
- **ACM Digital Library** - Computing research

## Response Format
When similar papers are found, the response includes:

### Paper Information
- **Title**: Full paper title
- **Source**: Database/platform where found
- **URL**: Direct link to paper
- **Summary**: Brief description of the research
- **Relevance Score**: How closely it matches your paper

### Comparative Analysis
- **Key similarities** with your paper
- **Important differences** in approach/methodology
- **Novel contributions** of each paper
- **Research trends** and developments

## Configuration Options

### In config.py:
```python
# Enable/disable internet search
ENABLE_INTERNET_SEARCH = True

# Maximum number of similar papers to find
MAX_SEARCH_RESULTS = 5

# Search timeout (seconds)
SEARCH_TIMEOUT = 30

# Domains to search
SEARCH_DOMAINS = [
    "arxiv.org",
    "scholar.google.com", 
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov"
]
```

## API Requirements

### Required:
- **GROQ_API_KEY**: For LLM functionality

### Optional (but recommended):
- **TAVILY_API_KEY**: Enables internet search
  - Without this, only local paper analysis works
  - Get free key at: https://tavily.com/

## Benefits

### For Researchers:
1. **Discover Related Work**: Automatically find papers you might have missed
2. **Stay Updated**: Find recent developments in your field
3. **Comparative Insights**: Understand how your work relates to others
4. **Citation Discovery**: Find additional papers to cite

### For Literature Reviews:
1. **Comprehensive Coverage**: Broader search than manual methods
2. **Real-time Results**: Access to latest published research
3. **Relevance Ranking**: Most relevant papers first
4. **Source Diversity**: Multiple academic databases

## Tips for Best Results

### Paper Titles:
- Use descriptive, technical titles for better search results
- Include key methodology or domain terms

### Questions:
- Be specific: "Recent deep learning papers for medical imaging"
- Use domain keywords: "transformer models", "computer vision", etc.
- Ask for comparisons: "How does this compare to recent work?"

### Search Optimization:
- The system automatically extracts key terms from your paper
- Questions with "recent", "latest", "current" search for newer papers
- Questions with "benchmark", "survey" find comprehensive studies

## Troubleshooting

### No Search Results:
- Check TAVILY_API_KEY in .env file
- Try more general search terms
- Verify internet connection

### Poor Results:
- Use more specific technical terms in questions
- Check if paper title contains clear technical keywords
- Try different phrasing for your question

### Search Not Triggered:
- Use keywords like "similar", "related", "recent"
- Try the Quick Action button
- Ask explicitly: "Find papers similar to this research"

## Future Enhancements

### Planned Features:
1. **Citation Network Analysis**: Map relationships between papers
2. **Author Tracking**: Find papers by same/similar authors  
3. **Trend Analysis**: Identify research trends over time
4. **Custom Filters**: Filter by publication date, venue, etc.
5. **Export Results**: Save similar papers list

---

**Note**: The Similar Papers feature requires an internet connection and works best with a Tavily API key. Without the API key, the system falls back to local paper analysis only.
