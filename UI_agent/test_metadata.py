#!/usr/bin/env python3
"""
Test script for the improved metadata extraction function
"""

from pdf_processor import extract_paper_metadata
import os

def test_metadata_extraction():
    """Test the improved metadata extraction with existing PDFs"""
    papers_folder = 'papers'
    
    if os.path.exists(papers_folder):
        pdf_files = [f for f in os.listdir(papers_folder) if f.endswith('.pdf')]
        print(f'Found {len(pdf_files)} PDF files in papers folder:')
        
        for pdf_file in pdf_files[:3]:  # Test first 3 PDFs
            pdf_path = os.path.join(papers_folder, pdf_file)
            print(f'\n--- Testing: {pdf_file} ---')
            
            try:
                metadata = extract_paper_metadata(pdf_path)
                print(f'Title: {metadata.get("title", "N/A")}')
                print(f'Authors: {metadata.get("authors", "N/A")}')
                print(f'Pages: {metadata.get("total_pages", "N/A")}')
                
                # Show if extraction was successful
                if metadata.get("title") != "Unknown Title":
                    print("✅ Title extraction: SUCCESS")
                else:
                    print("❌ Title extraction: FAILED")
                    
                if metadata.get("authors") != "Unknown Authors":
                    print("✅ Author extraction: SUCCESS")
                else:
                    print("❌ Author extraction: FAILED")
                    
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
    else:
        print('No papers folder found')
        # Test with PDFs in parent directory
        parent_dir = '..'
        pdf_files = []
        for file in os.listdir(parent_dir):
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(parent_dir, file))
        
        if pdf_files:
            print(f'Found {len(pdf_files)} PDF files in parent directory')
            for pdf_path in pdf_files[:2]:  # Test first 2 PDFs
                print(f'\n--- Testing: {os.path.basename(pdf_path)} ---')
                try:
                    metadata = extract_paper_metadata(pdf_path)
                    print(f'Title: {metadata.get("title", "N/A")}')
                    print(f'Authors: {metadata.get("authors", "N/A")}')
                except Exception as e:
                    print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Testing improved metadata extraction with font size analysis...")
    test_metadata_extraction()
