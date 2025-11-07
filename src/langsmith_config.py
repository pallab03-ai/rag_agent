import os
import logging
from typing import Optional

logger = logging.getLogger("rag_agent")


def setup_langsmith(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    enable_tracing: Optional[bool] = None
) -> bool:
    if api_key is None:
        api_key = os.getenv("LANGCHAIN_API_KEY")
    
    if project is None:
        project = os.getenv("LANGCHAIN_PROJECT", "rag-agent-langgraph")
    
    if enable_tracing is None:
        enable_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    if not enable_tracing:
        logger.info("LangSmith tracing is disabled")
        return False
    
    if not api_key or api_key == "your_langsmith_key_optional":
        logger.warning(
            "LangSmith tracing is enabled but no valid API key found. "
            "Set LANGCHAIN_API_KEY environment variable to enable tracing."
        )
        return False
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    logger.info(f"✓ LangSmith tracing enabled for project: {project}")
    logger.info(f"  View traces at: https://smith.langchain.com/o/default/projects/{project}")
    
    return True


def disable_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.info("LangSmith tracing disabled")


def get_langsmith_url(project: Optional[str] = None) -> Optional[str]:
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() != "true":
        return None
    
    if project is None:
        project = os.getenv("LANGCHAIN_PROJECT", "rag-agent-langgraph")
    
    return f"https://smith.langchain.com/o/default/projects/{project}"


def log_langsmith_info():
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "rag-agent-langgraph")
    
    if enabled:
        logger.info("="*80)
        logger.info("LANGSMITH TRACING")
        logger.info("="*80)
        logger.info(f"Status: ENABLED")
        logger.info(f"Project: {project}")
        logger.info(f"Dashboard: https://smith.langchain.com/o/default/projects/{project}")
        logger.info("="*80)
    else:
        logger.info("LangSmith tracing: DISABLED")
        logger.info("To enable: Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env")
