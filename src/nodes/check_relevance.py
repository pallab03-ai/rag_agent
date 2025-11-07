import logging

from ..state import AgentState
from ..utils import (
    log_node_execution,
    format_error_message,
    parse_yes_no_response,
    get_env_variable
)
from ..llm_factory import create_llm


logger = logging.getLogger("rag_agent")


def check_relevance_node(state: AgentState) -> AgentState:
    log_node_execution(logger, "CHECK_RELEVANCE", state)
    
    try:
        question = state["question"]
        retrieved_docs = state.get("retrieved_docs", [])
        context = state.get("context", "")
        
        # Check if we have documents to evaluate
        if not retrieved_docs or not context:
            logger.warning("No documents to evaluate relevance")
            state["is_relevant"] = False
            state["needs_web_search"] = True
            return state
        
        logger.info("Evaluating relevance of retrieved documents...")
        
        # Initialize LLM using factory
        llm = create_llm(temperature=0.0)  # Use low temperature for evaluation
        
        # Create relevance checking prompt
        relevance_prompt = f"""You are a relevance evaluator. Your task is to determine if the provided documents contain information that can answer the given question.

Question: {question}

Retrieved Documents:
{context[:2000]}  

Instructions:
- Analyze if the documents contain relevant information to answer the question
- Consider partial answers as relevant
- If documents discuss the topic but don't fully answer, still consider them relevant

Respond with ONLY "YES" or "NO":
- YES: If the documents contain information relevant to answering the question
- NO: If the documents are completely irrelevant or don't address the question

Your response (YES or NO):"""
        
        # Get LLM response
        logger.debug("Calling LLM for relevance check...")
        response = llm.invoke(relevance_prompt)
        response_text = response.content.strip()
        
        logger.info(f"Relevance check response: {response_text}")
        
        # Parse YES/NO response
        is_relevant = parse_yes_no_response(response_text)
        
        # Set flags based on relevance
        state["is_relevant"] = is_relevant
        state["needs_web_search"] = not is_relevant  # Need web search if not relevant
        
        if is_relevant:
            logger.info("✓ Documents are RELEVANT - proceeding to answer")
        else:
            logger.info("✗ Documents are NOT RELEVANT - will perform web search")
        
        logger.info("✓ Check relevance node completed successfully")
        
        return state
        
    except Exception as e:
        error_msg = format_error_message(e, "Check relevance node")
        logger.error(error_msg)
        state["error"] = error_msg
        # Default to web search on error
        state["is_relevant"] = False
        state["needs_web_search"] = True
        return state
