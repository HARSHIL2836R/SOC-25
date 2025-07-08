# Custom Semantic Chunking Implementation

## Overview

This update implements a **custom semantic chunking** solution as an alternative to the previous recursive character text splitter. Since the official LangChain SemanticChunker was not available in the current version, we built our own implementation that creates more coherent text segments by analyzing the semantic similarity between sentences using sentence embeddings and cosine similarity.

## What Changed

### 1. Dependencies
- Added `scikit-learn>=1.0.0` to requirements.txt for cosine similarity calculations
- Added `numpy>=1.21.0` for numerical operations
- Utilizes your existing `sentence-transformers` embeddings model

### 2. New Implementation (vector_store.py)
- **CustomSemanticChunker class**: A complete semantic chunking implementation
- **Sentence splitting**: Uses regex-based sentence detection
- **Similarity calculation**: Computes cosine similarity between consecutive sentences
- **Adaptive breakpoints**: Finds semantic boundaries using statistical thresholds
- **Buffer support**: Includes surrounding sentences for context preservation

### 2. Configuration (config.py)
New configuration options added:
```python
# Semantic chunking settings
SEMANTIC_CHUNKING_ENABLED: bool = True
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE: str = "percentile"  # "percentile", "standard_deviation", or "interquartile"
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: float = 95.0  # For percentile: 95th percentile
SEMANTIC_BUFFER_SIZE: int = 1  # Number of sentences to include on either side of a breakpoint
```

### 3. Vector Store (vector_store.py)
- Added `create_text_splitter()` helper function with fallback to recursive splitter
- Updated `create_vector_store()` to support both chunking methods
- Added `display_chunking_config()` for user visibility
- Enhanced error handling and logging

## How Our Custom Semantic Chunking Works

1. **Sentence Detection**: Text is split into sentences using regex patterns (`(?<=[.!?])\s+`)
2. **Embedding Generation**: Each sentence is embedded using your configured embeddings model
3. **Similarity Analysis**: Cosine similarity is calculated between consecutive sentence embeddings
4. **Statistical Thresholds**: Breakpoints are identified using configurable statistical methods:
   - **Percentile**: Chunks split when similarity falls below N-th percentile
   - **Standard Deviation**: Uses mean ± N standard deviations as threshold
   - **Interquartile Range**: Based on Q1 - N×IQR formula
5. **Buffer Addition**: Optional sentences added around breakpoints for context
6. **Chunk Assembly**: Text segments are combined respecting detected boundaries

## Implementation Details

### CustomSemanticChunker Class
```python
class CustomSemanticChunker:
    def __init__(self, embeddings, breakpoint_threshold_type="percentile", 
                 breakpoint_threshold_amount=95.0, buffer_size=1)
```

### Key Methods
- `_split_into_sentences()`: Regex-based sentence splitting
- `_calculate_similarities()`: Embedding-based similarity computation
- `_find_breakpoints()`: Statistical threshold analysis
- `_create_chunks_with_buffer()`: Context-aware chunk assembly
- `create_documents()`: Main interface returning Document objects

## Configuration Options

### Breakpoint Threshold Types
- **percentile** (recommended): Uses percentile-based thresholds (e.g., 95th percentile)
- **standard_deviation**: Uses standard deviation from the mean
- **interquartile**: Uses interquartile range for threshold calculation

### Threshold Amount
- For percentile: 95.0 means chunks are split when similarity is below the 95th percentile
- Higher values = fewer, larger chunks
- Lower values = more, smaller chunks

### Buffer Size
- Number of sentences to include on either side of detected breakpoints
- Helps maintain context across chunk boundaries
- Default: 1 sentence

## Benefits of Semantic Chunking

1. **Better Context Preservation**: Chunks are more likely to contain complete thoughts
2. **Improved Retrieval**: More relevant chunks for similarity search
3. **Adaptive Sizing**: Chunks naturally vary in size based on content structure
4. **Research Paper Friendly**: Works well with academic papers' logical structure

## Usage

The system automatically uses semantic chunking when `SEMANTIC_CHUNKING_ENABLED = True`. You can switch back to recursive chunking by setting it to `False`.

### Installation
Install required packages (handled automatically by the migration script):
```bash
pip install scikit-learn>=1.0.0 numpy>=1.21.0
```

Or run the migration script:
```bash
python migrate_to_semantic.py
```

### Testing
Verify the implementation with the test script:
```bash
python test_semantic_chunking.py
```

### Monitoring
The UI will display:
- Which chunking method is active
- Number of chunks created
- Configuration details in the expandable section

## Troubleshooting

### Import Errors
If you get import errors:
1. Run `pip install scikit-learn numpy`
2. Restart your Streamlit application
3. The system will automatically fall back to recursive chunking if semantic chunking fails

### Performance Considerations
- **Embedding Computation**: Semantic chunking requires embedding each sentence individually
- **Memory Usage**: Stores embeddings for similarity calculations
- **Processing Time**: Slower than recursive splitting, especially for long documents
- **Optimization**: Consider sentence batching for very large texts

### Quality Tuning
- **Threshold Amount**: Higher values (95-99) create fewer, larger chunks
- **Buffer Size**: Larger buffers provide more context but increase overlap
- **Threshold Type**: Percentile typically works best for research papers

## Fallback Behavior

The implementation includes robust fallback mechanisms:
1. If semantic chunker import fails → falls back to recursive splitter
2. If semantic chunking fails during processing → continues with recursive splitter
3. Configuration allows easy switching between methods

This ensures your application continues working even if there are issues with the semantic chunking setup.
