import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .nodes.plan import plan_node
from .nodes.retrieve import retrieve_node, set_vector_store_manager
from .nodes.check_relevance import check_relevance_node
from .nodes.web_search import web_search_node
from .nodes.answer import answer_node
from .nodes.reflect import reflect_node
from .routing import route_after_plan, route_after_check_relevance
from .tools.vector_store import VectorStoreManager
from .utils import setup_logging
from .langsmith_config import setup_langsmith, log_langsmith_info


logger = logging.getLogger("rag_agent")


class RAGAgent:
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        enable_reflection: bool = True,
        enable_langsmith: Optional[bool] = None
    ):
        if enable_langsmith is None:
            setup_langsmith()
        elif enable_langsmith:
            setup_langsmith(enable_tracing=True)
        
        log_langsmith_info()
        
        self.vector_store_manager = vector_store_manager
        self.enable_reflection = enable_reflection
        
        if vector_store_manager:
            set_vector_store_manager(vector_store_manager)
            logger.info("Vector store manager configured")
        
        self.graph = self._build_graph()
        
        logger.info("RAG Agent initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        logger.info("Building LangGraph workflow...")
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("plan", plan_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("check_relevance", check_relevance_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("answer", answer_node)
        
        if self.enable_reflection:
            workflow.add_node("reflect", reflect_node)
        
        workflow.set_entry_point("plan")
        
        workflow.add_conditional_edges(
            "plan",
            route_after_plan,
            {
                "retrieve": "retrieve",
                "answer": "answer"
            }
        )
        
        workflow.add_edge("retrieve", "check_relevance")
        
        workflow.add_conditional_edges(
            "check_relevance",
            route_after_check_relevance,
            {
                "web_search": "web_search",
                "answer": "answer"
            }
        )
        
        workflow.add_edge("web_search", "answer")
        
        if self.enable_reflection:
            workflow.add_edge("answer", "reflect")
            workflow.add_edge("reflect", END)
        else:
            workflow.add_edge("answer", END)
        
        compiled_graph = workflow.compile()
        
        logger.info("✓ LangGraph workflow built successfully")
        logger.info(f"Nodes: plan, retrieve, check_relevance, web_search, answer{', reflect' if self.enable_reflection else ''}")
        
        return compiled_graph
    
    def run(self, question: str) -> AgentState:
        logger.info("="*80)
        logger.info(f"Starting RAG Agent for question: '{question}'")
        logger.info("="*80)
        
        try:
            initial_state = create_initial_state(question)
            final_state = self.graph.invoke(initial_state)
            
            logger.info("="*80)
            logger.info("RAG Agent completed successfully")
            logger.info(f"Answer source: {final_state.get('source', 'unknown')}")
            logger.info("="*80)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Error running RAG Agent: {e}")
            # Return error state
            error_state = create_initial_state(question)
            error_state["error"] = str(e)
            error_state["answer"] = f"I apologize, but I encountered an error: {str(e)}"
            return error_state
    
    def stream(self, question: str):
        logger.info(f"Streaming RAG Agent for question: '{question}'")
        
        try:
            initial_state = create_initial_state(question)
            
            for state in self.graph.stream(initial_state):
                yield state
                
        except Exception as e:
            logger.error(f"Error streaming RAG Agent: {e}")
            error_state = create_initial_state(question)
            error_state["error"] = str(e)
            yield error_state


def create_rag_agent(
    vector_store_manager: Optional[VectorStoreManager] = None,
    enable_reflection: bool = True,
    log_level: str = "INFO"
) -> RAGAgent:
    setup_logging(log_level=log_level)
    
    agent = RAGAgent(
        vector_store_manager=vector_store_manager,
        enable_reflection=enable_reflection
    )
    
    return agent


def run_agent(question: str, vector_store_manager: Optional[VectorStoreManager] = None) -> dict:
    agent = create_rag_agent(vector_store_manager=vector_store_manager)
    final_state = agent.run(question)
    
    return {
        "question": final_state["question"],
        "answer": final_state.get("answer", "No answer generated"),
        "source": final_state.get("source", "unknown"),
        "reflection": final_state.get("reflection"),
        "error": final_state.get("error")
    }
