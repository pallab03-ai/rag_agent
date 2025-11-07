import logging
from typing import Literal

from .state import AgentState


logger = logging.getLogger("rag_agent")


def should_retrieve(state: AgentState) -> Literal["retrieve", "answer"]:
    needs_retrieval = state.get("needs_retrieval", False)
    
    if needs_retrieval:
        logger.info("Routing: PLAN → RETRIEVE (retrieval needed)")
        return "retrieve"
    else:
        logger.info("Routing: PLAN → ANSWER (no retrieval needed)")
        return "answer"


def should_web_search(state: AgentState) -> Literal["web_search", "answer"]:
    needs_web_search = state.get("needs_web_search", False)
    is_relevant = state.get("is_relevant", False)
    
    if needs_web_search or not is_relevant:
        logger.info("Routing: CHECK_RELEVANCE → WEB_SEARCH (documents not relevant)")
        state["source"] = "web"
        return "web_search"
    else:
        logger.info("Routing: CHECK_RELEVANCE → ANSWER (documents are relevant)")
        state["source"] = "local"
        return "answer"


def route_after_plan(state: AgentState) -> str:
    if state.get("error"):
        logger.warning(f"Error in state, routing to answer: {state['error']}")
        return "answer"
    
    return should_retrieve(state)


def route_after_check_relevance(state: AgentState) -> str:
    if state.get("error"):
        logger.warning(f"Error in state, routing to answer: {state['error']}")
        return "answer"
    
    return should_web_search(state)
