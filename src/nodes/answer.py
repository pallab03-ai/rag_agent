import logging

from ..state import AgentState
from ..utils import log_node_execution, format_error_message, get_env_variable
from ..llm_factory import create_llm


logger = logging.getLogger("rag_agent")


def answer_node(state: AgentState) -> AgentState:
    log_node_execution(logger, "ANSWER", state)
    
    try:
        question = state["question"]
        context = state.get("context", "")
        source = state.get("source", "none")
        
        logger.info(f"Generating answer for: '{question[:100]}...'")
        logger.info(f"Using context source: {source}")
        
        # Initialize LLM using factory
        llm = create_llm(
            temperature=float(get_env_variable("TEMPERATURE", "0.7")),
            max_tokens=int(get_env_variable("MAX_TOKENS", "1000"))
        )
        
        # Create answer prompt based on whether we have context
        if context and context.strip():
            # Answer with context
            if source == "web":
                source_label = "web search results"
            elif source == "local":
                source_label = "local knowledge base"
            else:
                source_label = "available information"
            
            answer_prompt = f"""You are a helpful AI assistant. Answer the user's question using the provided context.

Question: {question}

Context from {source_label}:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the context
- If the context doesn't fully answer the question, mention what information is available
- Be concise but informative
- Use natural language
- At the end, add a note about the source in this format:
  
  Source: [Local Knowledge Base] or [Web Search]

Your answer:"""
        else:
            # Answer without context (general knowledge)
            logger.info("No context available, answering from general knowledge")
            answer_prompt = f"""You are a helpful AI assistant. Answer the user's question based on your general knowledge.

Question: {question}

Instructions:
- Provide a helpful answer based on your training
- Be honest if you don't have enough information
- Be concise but informative
- At the end, add: "Source: [General Knowledge]"

Your answer:"""
            source = "none"
        
        # Get LLM response
        logger.debug("Calling LLM for answer generation...")
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
        
        logger.info(f"Answer generated ({len(answer)} characters)")
        logger.debug(f"Answer preview: {answer[:200]}...")
        
        # Update state
        state["answer"] = answer
        
        # Set source if not already set
        if not state.get("source"):
            if state.get("retrieved_docs"):
                state["source"] = "local"
            elif state.get("web_search_results"):
                state["source"] = "web"
            else:
                state["source"] = "none"
        
        logger.info(f"✓ Answer node completed successfully (source: {state['source']})")
        
        return state
        
    except Exception as e:
        error_msg = format_error_message(e, "Answer node")
        logger.error(error_msg)
        state["error"] = error_msg
        # Provide fallback answer
        state["answer"] = f"I apologize, but I encountered an error while generating the answer: {str(e)}"
        state["source"] = "error"
        return state
