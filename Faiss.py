import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# Step 1: Load environment variables
load_dotenv()
openai_key: str | None = os.getenv("OPENAI_API_KEY")

# Step 2: Function to load and parse PDF
def parse_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Step 3: Chunk the text
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([text])

# Step 4: Embed and store in FAISS
def embed_and_store(docs: list[Document], faiss_index_path: str = "faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    return vectorstore

# Entry point
if __name__ == "__main__":
    # Replace this with your PDF file path
    pdf_path = "example_paper.pdf"

    print("Parsing PDF...")
    raw_text = parse_pdf(pdf_path)

    print("Chunking text...")
    documents = chunk_text(raw_text)

    print("Embedding and storing in FAISS...")
    vectorstore = embed_and_store(documents)

    print("✅ Successfully processed and stored in FAISS!")
