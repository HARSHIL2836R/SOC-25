"""
LangGraph-based research agent with internet search capabilities for finding similar papers.
"""

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Any, Dict, Optional, List, TypedDict, Annotated
import json
import re
from datetime import datetime

# Import search capabilities
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from config import (
    SIMILARITY_SEARCH_K, 
    TAVILY_API_KEY, 
    MAX_SEARCH_RESULTS, 
    SEARCH_TIMEOUT,
    ENABLE_INTERNET_SEARCH,
    SEARCH_DOMAINS
)

class AgentState(TypedDict):
    """State definition for the research agent"""
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    paper_title: str
    paper_authors: str
    chat_history: str
    context: str
    similar_papers: List[Dict[str, Any]]
    search_performed: bool
    current_task: str

class ResearchAgent:
    """LangGraph-based research agent with internet search capabilities"""
    
    def __init__(self, vector_store: Any, llm: Any, metadata: Dict[str, Any]):
        self.vector_store = vector_store
        self.llm = llm
        self.metadata = metadata
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SIMILARITY_SEARCH_K}
        )
        
        # Initialize Tavily client for internet search
        self.tavily_client = None
        if ENABLE_INTERNET_SEARCH and TAVILY_API_KEY and TavilyClient:
            try:
                self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            except Exception as e:
                st.warning(f"Could not initialize Tavily search: {e}")
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph state graph for the research agent"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_question)
        workflow.add_node("local_search", self._local_paper_search)
        workflow.add_node("internet_search", self._internet_search)
        workflow.add_node("response_generator", self._generate_response)
        
        # Define the graph flow
        workflow.set_entry_point("router")
        
        # Router decides between local search only or internet search
        workflow.add_conditional_edges(
            "router",
            self._should_search_internet,
            {
                "internet_search": "internet_search",
                "local_search": "local_search"
            }
        )
        
        # Both paths lead to response generation
        workflow.add_edge("local_search", "response_generator")
        workflow.add_edge("internet_search", "response_generator")
        workflow.add_edge("response_generator", END)
        
        return workflow.compile()
    
    def _route_question(self, state: AgentState) -> AgentState:
        """Route the question and extract relevant context from the local paper"""
        try:
            question = state["question"]
            
            # Get context from local paper
            retrieved_docs = self.retriever.invoke(question)
            context = self._format_research_docs(retrieved_docs)
            
            # Update state
            state["context"] = context
            state["current_task"] = "routing"
            
            return state
        except Exception as e:
            st.error(f"Error in routing: {e}")
            state["context"] = ""
            return state
    
    def _should_search_internet(self, state: AgentState) -> str:
        """Determine if internet search is needed based on question keywords"""
        question = state["question"].lower()
        
        # Keywords that indicate need for similar papers
        search_keywords = [
            "similar", "related", "comparable", "like this", "other papers",
            "recent work", "state of art", "compare", "benchmark", "survey",
            "recent research", "latest", "current", "trends", "developments",
            "comparative analysis", "find papers", "search papers"
        ]
        
        # Check if question asks for similar papers or related work
        if any(keyword in question for keyword in search_keywords):
            return "internet_search"
        else:
            return "local_search"
    
    def _local_paper_search(self, state: AgentState) -> AgentState:
        """Search only in the local paper without internet search"""
        state["search_performed"] = False
        state["similar_papers"] = []
        state["current_task"] = "local_analysis"
        return state
    
    def _internet_search(self, state: AgentState) -> AgentState:
        """Search the internet for similar papers"""
        try:
            if not self.tavily_client:
                st.warning("Internet search not available. Using local paper only.")
                return self._local_paper_search(state)
            
            # Extract key terms from the paper for search
            paper_title = self.metadata.get('title', '')
            paper_authors = self.metadata.get('authors', '')
            
            # Create search query
            search_query = self._optimize_search_query_for_similar_papers(paper_title, state["question"])
            
            # Perform search
            st.info(f"🔍 Searching for similar papers: {search_query}")
            
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=MAX_SEARCH_RESULTS,
                include_domains=SEARCH_DOMAINS
            )
            
            # Process and filter results
            similar_papers = self._process_search_results(search_results)
            
            state["similar_papers"] = similar_papers
            state["search_performed"] = True
            state["current_task"] = "internet_search_complete"
            
            if similar_papers:
                st.success(f"✅ Found {len(similar_papers)} similar papers")
            else:
                st.info("No similar papers found in search results")
            
            return state
            
        except Exception as e:
            st.error(f"Error during internet search: {e}")
            state["similar_papers"] = []
            state["search_performed"] = False
            return state
    
    def _optimize_search_query_for_similar_papers(self, paper_title: str, question: str) -> str:
        """Create an optimized search query specifically for finding similar papers"""
        # Extract key technical terms from title
        title_words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{3,}\b', paper_title)
        
        # Filter out common words
        stop_words = {'the', 'and', 'for', 'with', 'using', 'based', 'from', 'into', 'through'}
        key_terms = [word for word in title_words if len(word) > 3 and word.lower() not in stop_words][:6]
        
        # Combine with domain-specific search terms
        search_terms = " ".join(key_terms)
        
        # Add research-specific modifiers based on question intent
        if any(word in question.lower() for word in ["recent", "latest", "current"]):
            search_query = f"{search_terms} recent research papers 2023 2024 2025"
        elif any(word in question.lower() for word in ["comparative", "compare", "benchmark"]):
            search_query = f"{search_terms} comparative analysis benchmark study"
        elif any(word in question.lower() for word in ["survey", "review"]):
            search_query = f"{search_terms} survey review state of the art"
        else:
            search_query = f"{search_terms} similar research methodology approach"
        
        return search_query

    def _process_search_results(self, search_results: Dict) -> List[Dict[str, Any]]:
        """Process and filter search results to extract paper information"""
        papers = []
        
        try:
            results = search_results.get('results', [])
            
            for result in results:
                url = result.get('url', '')
                title = result.get('title', '')
                content = result.get('content', '')
                
                # Filter for academic papers
                if self._is_academic_paper(url, title, content):
                    paper_info = {
                        'title': title,
                        'url': url,
                        'summary': content[:500] + "..." if len(content) > 500 else content,
                        'source': self._extract_source(url),
                        'relevance_score': result.get('score', 0)
                    }
                    papers.append(paper_info)
            
            # Sort by relevance score
            papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
        except Exception as e:
            st.error(f"Error processing search results: {e}")
        
        return papers[:MAX_SEARCH_RESULTS]
    
    def _is_academic_paper(self, url: str, title: str, content: str) -> bool:
        """Check if the result is likely an academic paper"""
        academic_indicators = [
            'arxiv.org', 'scholar.google', 'semanticscholar.org', 
            'pubmed.ncbi.nlm.nih.gov', 'ieee.org', 'acm.org',
            'springer.com', 'nature.com', 'science.org'
        ]
        
        # Check URL
        url_is_academic = any(indicator in url.lower() for indicator in academic_indicators)
        
        # Check title for academic patterns
        title_indicators = [':', 'analysis', 'study', 'research', 'model', 'algorithm']
        title_is_academic = any(indicator in title.lower() for indicator in title_indicators)
        
        return url_is_academic or title_is_academic
    
    def _extract_source(self, url: str) -> str:
        """Extract source name from URL"""
        if 'arxiv.org' in url:
            return 'arXiv'
        elif 'scholar.google' in url:
            return 'Google Scholar'
        elif 'semanticscholar.org' in url:
            return 'Semantic Scholar'
        elif 'pubmed' in url:
            return 'PubMed'
        elif 'ieee.org' in url:
            return 'IEEE'
        elif 'acm.org' in url:
            return 'ACM'
        else:
            return 'Academic'
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response using the gathered information"""
        try:
            # Create prompt based on whether internet search was performed
            if state["search_performed"] and state["similar_papers"]:
                prompt = self._create_enhanced_prompt()
            else:
                prompt = self._create_basic_prompt()
            
            # Prepare input variables
            prompt_inputs = {
                'context': state["context"],
                'question': state["question"],
                'chat_history': state["chat_history"],
                'paper_title': state["paper_title"],
                'authors': state["paper_authors"]
            }
            
            # Add similar papers if available
            if state["similar_papers"]:
                similar_papers_text = self._format_similar_papers(state["similar_papers"])
                prompt_inputs['similar_papers'] = similar_papers_text
            
            # Generate response
            parser = StrOutputParser()
            chain = prompt | self.llm | parser
            response = chain.invoke(prompt_inputs)
            
            # Create AI message
            ai_message = AIMessage(content=response)
            state["messages"] = [ai_message]
            
            return state
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            error_response = f"I apologize, but I encountered an error while processing your question: {e}"
            ai_message = AIMessage(content=error_response)
            state["messages"] = [ai_message]
            return state
    
    def _create_basic_prompt(self) -> PromptTemplate:
        """Create basic prompt for local paper analysis only"""
        return PromptTemplate(
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

CONVERSATION HISTORY:
{chat_history}

RELEVANT PAPER CONTENT:
{context}

RESEARCH QUESTION: {question}

ANSWER:""",
            input_variables=['context', 'question', 'chat_history', 'paper_title', 'authors']
        )
    
    def _create_enhanced_prompt(self) -> PromptTemplate:
        """Create enhanced prompt that includes similar papers from internet search"""
        return PromptTemplate(
            template="""You are an expert research assistant specializing in academic paper analysis with access to current research.

PAPER INFORMATION:
Title: {paper_title}
Authors: {authors}

INSTRUCTIONS:
- Analyze the provided research paper content to answer questions accurately
- Reference specific sections, methodologies, results, and findings when relevant
- When relevant, incorporate information from similar papers found through internet search
- Compare and contrast with related work when appropriate
- Maintain academic rigor and cite evidence from both the main paper and related work
- Use the conversation history to provide coherent, contextual responses

CONVERSATION HISTORY:
{chat_history}

RELEVANT PAPER CONTENT:
{context}

SIMILAR PAPERS FROM RECENT SEARCH:
{similar_papers}

RESEARCH QUESTION: {question}

ANSWER:""",
            input_variables=['context', 'question', 'chat_history', 'paper_title', 'authors', 'similar_papers']
        )
    
    def _format_similar_papers(self, papers: List[Dict[str, Any]]) -> str:
        """Format similar papers for inclusion in prompt"""
        if not papers:
            return "No similar papers found."
        
        formatted = []
        for i, paper in enumerate(papers, 1):
            formatted.append(f"""
📄 **Paper {i}:**
🔷 **Title:** {paper['title']}
🏛️ **Source:** {paper['source']}
🔗 **URL:** {paper['url']}
📋 **Summary:** {paper['summary']}
⭐ **Relevance Score:** {paper.get('relevance_score', 'N/A')}
""")
        
        return "\n" + "="*60 + "\n".join(formatted)
    
    def _format_research_docs(self, retrieved_docs) -> str:
        """Format retrieved documents from the local paper"""
        formatted_content = []
        for doc in retrieved_docs:
            section = doc.metadata.get('section', 'Unknown Section')
            content = doc.page_content
            formatted_content.append(f"[Section: {section}]\n{content}")
        return "\n\n" + "="*50 + "\n\n".join(formatted_content)
    
    def process_question(self, question: str, chat_history: str = "") -> str:
        """Process a question using the LangGraph agent"""
        try:
            # Prepare initial state
            initial_state = AgentState(
                messages=[HumanMessage(content=question)],
                question=question,
                paper_title=self.metadata.get('title', 'Unknown Title'),
                paper_authors=self.metadata.get('authors', 'Unknown Authors'),
                chat_history=chat_history,
                context="",
                similar_papers=[],
                search_performed=False,
                current_task="starting"
            )
            
            # Run the agent
            final_state = self.graph.invoke(initial_state)
            
            # Extract response
            if final_state["messages"]:
                return final_state["messages"][-1].content
            else:
                return "I apologize, but I couldn't generate a response to your question."
                
        except Exception as e:
            st.error(f"Error in agent processing: {e}")
            return f"Error processing your question: {e}"

def setup_research_chain(vector_store: Any, llm: Any, metadata: Dict[str, Any]) -> Optional[ResearchAgent]:
    """Setup the LangGraph research agent"""
    try:
        agent = ResearchAgent(vector_store, llm, metadata)
        return agent
    except Exception as e:
        st.error(f"Error setting up research agent: {e}")
        return None

def process_question(research_agent: ResearchAgent, research_memory: Any, question: str) -> str:
    """Process a question using the research agent"""
    try:
        # Get chat history from memory
        chat_history = research_memory.load_memory_variables({})["chat_history"]
        if isinstance(chat_history, list):
            chat_history = "\n".join(str(x) for x in chat_history)
        elif chat_history is None:
            chat_history = ""
        
        # Get response from research agent
        answer = research_agent.process_question(question, chat_history)
        
        # Save to memory
        research_memory.save_context(
            {"question": question}, 
            {"answer": answer}
        )
        
        return answer
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return f"Error processing your question: {e}"
