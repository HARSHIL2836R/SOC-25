"""
PDF processing utilities for the Research Paper Analysis Agent.
"""

import os
import re
import urllib.request
from urllib.parse import urlparse
import pymupdf
import streamlit as st
from config import (
    SECTION_PATTERNS, MIN_SECTION_LENGTH, PAPERS_FOLDER,
    ARXIV_ABSTRACT_PATTERN, ARXIV_PDF_BASE, ensure_papers_folder
)
from typing import Any, Dict, List, Optional, Tuple, Union

def extract_paper_sections(pdf_path: str) -> Dict[str, str]:
    """Extract sections from a research paper PDF"""
    try:
        doc: pymupdf.Document = pymupdf.open(pdf_path)
        
        sections: Dict[str, str] = {}
        current_section: str = "content"
        section_content: List[str] = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                is_section_header = False
                for pattern in SECTION_PATTERNS:
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

def extract_paper_metadata(pdf_path: str) -> Dict[str, Union[str, int]]:
    """Extract basic metadata from research paper using font size analysis"""
    try:
        doc: pymupdf.Document = pymupdf.open(pdf_path)
        
        title: str = "Unknown Title"
        authors: str = "Unknown Authors"
        
        # Extract text with font information from first few pages
        text_blocks = []
        
        # Check first 2 pages for title and author information
        for page_num in range(min(2, doc.page_count)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            
                            # Filter out very short text, page numbers, headers, and metadata
                            if (len(text) > 10 and 
                                not text.isdigit() and 
                                not text.lower().startswith(('page', 'figure', 'table', 'doi:', 'http', 'www')) and
                                not re.match(r'^arxiv:\d+\.\d+', text.lower()) and  # Skip arXiv IDs
                                not re.match(r'^\[.*\]', text) and  # Skip bracketed metadata
                                bbox[1] > 50):  # Skip very top headers (y-coordinate > 50)
                                
                                text_blocks.append({
                                    'text': text,
                                    'font_size': font_size,
                                    'is_bold': bool(font_flags & 2**4),  # Bold flag
                                    'page': page_num,
                                    'y_position': bbox[1],  # Vertical position
                                    'line_height': bbox[3] - bbox[1]
                                })
        
        if text_blocks:
            # Sort by font size (descending) to find largest text
            text_blocks.sort(key=lambda x: x['font_size'], reverse=True)
            
            # Find the title - usually the largest font on the first page
            title_candidates = []
            max_font_size = text_blocks[0]['font_size']
            
            # Consider text within 15% of max font size as potential titles
            title_threshold = max_font_size * 0.85
            
            for block in text_blocks:
                if (block['font_size'] >= title_threshold and 
                    block['page'] == 0 and  # Title should be on first page
                    len(block['text']) > 15 and  # Reasonable title length
                    len(block['text']) < 200 and  # Not too long (avoid abstracts)
                    not block['text'].lower().startswith(('abstract', 'introduction', 'keywords')) and
                    # Avoid common academic formatting
                    not re.match(r'^\d+\s', block['text']) and  # Not starting with numbers
                    not block['text'].isupper()):  # Not all caps (usually headers)
                    
                    # Calculate title score based on multiple factors
                    score = 0
                    
                    # Font size score (larger = better)
                    score += (block['font_size'] / max_font_size) * 40
                    
                    # Position score (higher on page = better for title)
                    if block['y_position'] < 200:  # Top portion of page
                        score += 30
                    
                    # Length score (reasonable title length)
                    text_len = len(block['text'])
                    if 20 <= text_len <= 100:  # Ideal title length
                        score += 20
                    elif 15 <= text_len <= 150:  # Acceptable title length
                        score += 10
                    
                    # Bold text gets bonus points
                    if block['is_bold']:
                        score += 15
                    
                    # Penalize certain patterns
                    text_lower = block['text'].lower()
                    if any(word in text_lower for word in 
                           ['conference', 'proceedings', 'journal', 'workshop', 'symposium']):
                        score -= 20
                    
                    # Bonus for academic title patterns
                    if any(word in text_lower for word in 
                           ['analysis', 'study', 'approach', 'method', 'model', 'algorithm', 
                            'framework', 'system', 'learning', 'neural', 'deep']):
                        score += 10
                    
                    title_candidates.append({**block, 'score': score})
            
            if title_candidates:
                # Choose the highest scoring title candidate
                title_candidates.sort(key=lambda x: x['score'], reverse=True)
                title_block = title_candidates[0]
                title = title_block['text']
                
                # Clean up the title
                title = re.sub(r'\s+', ' ', title).strip()
                title = title.replace('\n', ' ').strip()
            
            # Find authors - look for text with author indicators
            author_candidates = []
            author_font_threshold = max_font_size * 0.7  # Authors usually smaller than title
            
            for block in text_blocks:
                text_lower = block['text'].lower()
                if (block['page'] == 0 and  # Authors usually on first page
                    block['font_size'] < author_font_threshold and
                    block['font_size'] > 8 and  # Not too small
                    len(block['text']) > 10 and
                    block['y_position'] > 100):  # Not in header area
                    
                    # Check for author indicators
                    has_institution = any(indicator in text_lower for indicator in 
                                        ['university', 'institute', 'college', 'dept', 'department', 
                                         'school', 'center', 'laboratory', '@'])
                    
                    # Check for name patterns
                    name_patterns = len(re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', block['text']))
                    
                    # Check for email patterns
                    has_email = '@' in block['text'] and '.' in block['text']
                    
                    if has_institution or name_patterns >= 1 or has_email:
                        score = 0
                        if has_institution:
                            score += 20
                        if has_email:
                            score += 15
                        score += name_patterns * 10
                        
                        # Prefer text that's not too close to title position
                        if title != "Unknown Title":
                            # Find title position
                            title_y = next((c['y_position'] for c in title_candidates 
                                          if c['text'] == title), 0)
                            if block['y_position'] > title_y + 50:  # Below title
                                score += 10
                        
                        author_candidates.append({**block, 'score': score})
            
            if author_candidates:
                # Choose the highest scoring author candidate
                author_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_author = author_candidates[0]
                authors = best_author['text']
                authors = re.sub(r'\s+', ' ', authors).strip()
        
        # Fallback to simple text extraction if font-based method fails
        if title == "Unknown Title":
            first_page_text = doc[0].get_text()
            lines = first_page_text.split('\n')
            substantial_lines = [line.strip() for line in lines if len(line.strip()) > 15]
            
            # Skip obvious metadata lines
            for line in substantial_lines:
                if (not re.match(r'^arxiv:\d+\.\d+', line.lower()) and
                    not line.lower().startswith(('abstract', 'keywords', 'doi:', 'http')) and
                    not re.match(r'^\[.*\]', line)):
                    title = line
                    title = re.sub(r'\s+', ' ', title).strip()
                    break
        
        metadata: Dict[str, Union[str, int]] = {
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

def process_arxiv_url(arxiv_url: str) -> str:
    """Convert arXiv abstract URL to PDF URL"""
    if ARXIV_ABSTRACT_PATTERN in arxiv_url:
        paper_id = arxiv_url.split('/abs/')[-1]
        pdf_url = ARXIV_PDF_BASE.format(paper_id)
        return pdf_url
    return arxiv_url

def extract_paper_from_url(url: str) -> Tuple[Dict[str, str], Dict[str, Union[str, int]]]:
    """Extract paper content from URL and save to papers folder"""
    try:
        if 'arxiv.org' in url:
            url = process_arxiv_url(url)
        
        # Create papers folder if it doesn't exist
        papers_folder: str = ensure_papers_folder()
        
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
        file_path: str = os.path.join(papers_folder, filename)
        
        # Download and save the file
        st.info(f"📥 Downloading: {filename}")
        urllib.request.urlretrieve(url, file_path)
        st.success(f"✅ Saved to: {file_path}")
        
        # Extract content from saved file
        sections: Dict[str, str] = extract_paper_sections(file_path)
        metadata: Dict[str, Union[str, int]] = extract_paper_metadata(file_path)
        metadata['source_url'] = url
        metadata['local_path'] = file_path
        
        return sections, metadata
    except Exception as e:
        st.error(f"Error extracting paper from URL: {e}")
        return {}, {}

def save_uploaded_file(uploaded_file: Any) -> Optional[str]:
    """Save uploaded file to papers folder and return file path"""
    try:
        papers_folder: str = ensure_papers_folder()
        file_path: str = os.path.join(papers_folder, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Saved to: {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def get_saved_papers() -> List[str]:
    """Get list of saved PDF files"""
    papers_folder: str = ensure_papers_folder()
    if os.path.exists(papers_folder):
        return [f for f in os.listdir(papers_folder) if f.endswith('.pdf')]
    return []

def delete_paper(filename: str) -> bool:
    """Delete a saved paper file"""
    try:
        file_path: str = os.path.join(PAPERS_FOLDER, filename)
        if os.path.exists(file_path):
            os.unlink(file_path)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False
