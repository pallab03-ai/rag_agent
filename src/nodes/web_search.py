import logging

from ..state import AgentState
from ..utils import log_node_execution, format_error_message, format_web_results
from ..tools.web_search_tool import create_web_search_tool


logger = logging.getLogger("rag_agent")


def web_search_node(state: AgentState) -> AgentState:
    log_node_execution(logger, "WEB_SEARCH", state)
    
    try:
        question = state["question"]
        
        logger.info(f"Performing web search for: '{question[:100]}...'")
        
        # Create web search tool
        search_tool = create_web_search_tool()
        
        # Perform search
        results = search_tool.search(query=question, use_fallback=True)
        
        if not results:
            logger.warning("No web search results found")
            state["web_search_results"] = []
            state["context"] = None
            state["source"] = "none"
            return state
        
        logger.info(f"Found {len(results)} web search results")
        
        # Log result snippets
        for i, result in enumerate(results, 1):
            logger.debug(f"Result {i}: {result.get('title', 'No title')[:50]}...")
        
        # Format results as context
        context = format_web_results(results)
        
        # Store raw results for reference
        web_results_formatted = []
        for result in results:
            web_results_formatted.append(
                f"Title: {result.get('title', 'No title')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Content: {result.get('content', '')[:300]}..."
            )
        
        # Update state
        state["web_search_results"] = web_results_formatted
        state["context"] = context
        state["source"] = "web"
        
        logger.info("✓ Web search node completed successfully")
        
        return state
        
    except Exception as e:
        error_msg = format_error_message(e, "Web search node")
        logger.error(error_msg)
        state["error"] = error_msg
        state["web_search_results"] = []
        state["context"] = None
        state["source"] = "none"
        return state
