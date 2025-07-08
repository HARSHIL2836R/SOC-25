"""
Research chain setup and management for the Research Paper Analysis Agent.
"""

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from config import SIMILARITY_SEARCH_K
from typing import Any, Dict, Optional

def setup_research_chain(vector_store: Any, llm: Any, metadata: Dict[str, Any]) -> Optional[Any]:
    """Setup the research chain"""
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SIMILARITY_SEARCH_K}
        )
        
        research_prompt = PromptTemplate(
            template="""You are an expert research assistant specializing in academic paper analysis.

PAPER INFORMATION:
Title: {paper_title}
Authors: {authors}

INSTRUCTIONS:
- Analyze the provided research paper content to answer questions accurately
- Reference specific sections, methodologies, results, and findings when relevant
- Maintain academic rigor and cite evidence from the paper
- Use the conversation history to provide coherent, contextual responses
- If the question requires information not in the provided context, clearly state the limitations
- For technical questions, explain concepts clearly while maintaining accuracy

CONVERSATION HISTORY:
{chat_history}

RELEVANT PAPER CONTENT:
{context}

RESEARCH QUESTION: {question}

ANSWER:""",
            input_variables=['context', 'question', 'chat_history', 'paper_title', 'authors']
        )
        
        def format_research_docs(retrieved_docs):
            formatted_content = []
            for doc in retrieved_docs:
                section = doc.metadata.get('section', 'Unknown Section')
                content = doc.page_content
                formatted_content.append(f"[Section: {section}]\n{content}")
            return "\n\n" + "="*50 + "\n\n".join(formatted_content)
        
        research_parallel_chain = RunnableParallel({
            'context': lambda inputs: format_research_docs(retriever.invoke(inputs.get('question', ''))),
            'question': lambda inputs: inputs.get('question', ''),
            'chat_history': lambda inputs: inputs.get('chat_history', ''),
            'paper_title': lambda inputs: metadata.get('title', 'Unknown Title'),
            'authors': lambda inputs: metadata.get('authors', 'Unknown Authors')
        })
        
        parser = StrOutputParser()
        research_chain = research_parallel_chain | research_prompt | llm | parser
        
        return research_chain
    except Exception as e:
        st.error(f"Error setting up research chain: {e}")
        return None

def process_question(research_chain: Any, research_memory: Any, question: str) -> str:
    """Process a question using the research chain"""
    try:
        # Get chat history from memory
        chat_history = research_memory.load_memory_variables({})["chat_history"]
        if isinstance(chat_history, list):
            chat_history = "\n".join(str(x) for x in chat_history)
        elif chat_history is None:
            chat_history = ""
        
        # Get response from research chain
        result = research_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        # Extract answer
        if isinstance(result, dict) and "answer" in result:
            answer = result["answer"]
        else:
            answer = result
        
        # Save to memory
        research_memory.save_context(
            {"question": question}, 
            {"answer": answer}
        )
        
        return answer
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return f"Error processing your question: {e}"
