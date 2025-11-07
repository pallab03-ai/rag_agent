import logging
from typing import Optional

from ..state import AgentState
from ..utils import log_node_execution, format_error_message, format_documents
from ..tools.vector_store import VectorStoreManager


logger = logging.getLogger("rag_agent")


_vector_store_manager: Optional[VectorStoreManager] = None


def set_vector_store_manager(manager: VectorStoreManager) -> None:
    global _vector_store_manager
    _vector_store_manager = manager
    logger.info("Vector store manager set for retrieve node")


def retrieve_node(state: AgentState) -> AgentState:
    log_node_execution(logger, "RETRIEVE", state)
    
    try:
        question = state["question"]
        
        # Check if vector store is available
        if _vector_store_manager is None:
            logger.error("Vector store manager not initialized")
            state["error"] = "Vector store not available"
            state["retrieved_docs"] = []
            state["context"] = None
            return state
        
        logger.info(f"Retrieving documents for: '{question[:100]}...'")
        
        # Get top_k from config or use default
        top_k = 5  # Default value
        
        # Perform similarity search
        results = _vector_store_manager.similarity_search(
            query=question,
            k=top_k
        )
        
        if not results:
            logger.warning("No documents retrieved from vector store")
            state["retrieved_docs"] = []
            state["context"] = None
            return state
        
        # Extract document contents
        retrieved_docs = [doc.page_content for doc in results]
        
        logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Log document snippets for debugging
        for i, doc in enumerate(retrieved_docs, 1):
            logger.debug(f"Doc {i} preview: {doc[:100]}...")
        
        # Format documents as context
        context = format_documents(retrieved_docs)
        
        # Update state
        state["retrieved_docs"] = retrieved_docs
        state["context"] = context
        state["source"] = "local"  # Mark as using local knowledge base
        
        logger.info("✓ Retrieve node completed successfully")
        
        return state
        
    except Exception as e:
        error_msg = format_error_message(e, "Retrieve node")
        logger.error(error_msg)
        state["error"] = error_msg
        state["retrieved_docs"] = []
        state["context"] = None
        return state
