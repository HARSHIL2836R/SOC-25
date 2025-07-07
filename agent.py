import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class ResearchAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the following context to answer the question: {context} \n If the question is not given in context, just say no.\n Question:",
                ),
                ("human", "{question}"),
            ]
        )

    def answer(self, path: str, question: str) -> str:
        doc = pymupdf.open(path)
        text = ""
        for page in doc:
            text += page.get_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=5,
            length_function=len,
        )
        docs = text_splitter.create_documents([text])

        vector_db = FAISS.from_texts(
            texts=[doc.page_content for doc in docs],
            embedding=self.embeddings,
            metadatas=[doc.metadata for doc in docs],
        )
        
        retriever = vector_db.as_retriever()
        context_docs = retriever.get_relevant_documents(question)
        
        context = ""
        for doc in context_docs:
            context += doc.page_content + "\n"


        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return response.content
