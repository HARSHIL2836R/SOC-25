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
    """Extract basic metadata from research paper"""
    try:
        doc: pymupdf.Document = pymupdf.open(pdf_path)
        
        first_page: str = doc[0].get_text()
        lines: List[str] = first_page.split('\n')
        
        title: str = "Unknown Title"
        authors: str = "Unknown Authors"
        
        substantial_lines: List[str] = [line.strip() for line in lines if len(line.strip()) > 10]
        if substantial_lines:
            title = substantial_lines[0]
            if len(substantial_lines) > 1:
                for line in substantial_lines[1:4]:
                    if any(indicator in line.lower() for indicator in ['@', 'university', 'institute', 'college']):
                        authors = line
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
