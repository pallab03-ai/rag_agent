import os
import logging
import yaml
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: str = "logs/agent_traces.log"
) -> logging.Logger:
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("rag_agent")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    base_path = Path(__file__).parent.parent / "configs"
    
    configs = {
        "agent": load_config(base_path / "agent_config.yaml"),
        "retrieval": load_config(base_path / "retrieval_config.yaml"),
        "web_search": load_config(base_path / "web_search_config.yaml")
    }
    
    return configs


def get_env_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(var_name, default)


def parse_yes_no_response(text: str) -> bool:
    text_upper = text.upper().strip()
    
    if "YES" in text_upper:
        return True
    
    if "NO" in text_upper:
        return False
    
    return False


def extract_confidence_score(text: str) -> float:
    patterns = [
        r'confidence[:\s]+([0-9]*\.?[0-9]+)',
        r'score[:\s]+([0-9]*\.?[0-9]+)',
        r'([0-9]*\.?[0-9]+)%',
        r'\b([0-9]*\.?[0-9]+)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            score = float(match.group(1))
            if '%' in text[match.start():match.end()]:
                score = score / 100.0
            return max(0.0, min(1.0, score))
    
    return 0.5


def format_documents(docs: List[str], max_length: Optional[int] = None) -> str:
    if not docs:
        return ""
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"Document {i}:\n{doc}\n")
    
    context = "\n".join(formatted)
    
    if max_length and len(context) > max_length:
        context = context[:max_length] + "...[truncated]"
    
    return context


def format_web_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', '')
        content = result.get('content', '') or result.get('snippet', '')
        
        formatted.append(f"Result {i}:\nTitle: {title}\nURL: {url}\nContent: {content}\n")
    
    return "\n".join(formatted)


def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_error_message(error: Exception, context: str = "") -> str:
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"{context} - {error_type}: {error_msg}"
    return f"{error_type}: {error_msg}"


def sanitize_question(question: str) -> str:
    question = " ".join(question.split())
    question = question.strip()
    question = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', question)
    
    return question


def is_question_valid(question: str, min_length: int = 3) -> bool:
    if not question or not question.strip():
        return False
    
    if len(question.strip()) < min_length:
        return False
    
    return True


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def log_node_execution(logger: logging.Logger, node_name: str, state: Dict[str, Any]) -> None:
    logger.info(f"=== Executing Node: {node_name} ===")
    logger.debug(f"State before {node_name}: {sanitize_state_for_logging(state)}")


def sanitize_state_for_logging(state: Dict[str, Any], max_length: int = 200) -> Dict[str, Any]:
    sanitized = {}
    for key, value in state.items():
        if isinstance(value, str) and len(value) > max_length:
            sanitized[key] = value[:max_length] + "...[truncated]"
        elif isinstance(value, list) and len(value) > 3:
            sanitized[key] = f"[List with {len(value)} items]"
        elif isinstance(value, dict):
            sanitized[key] = "[Dictionary]"
        else:
            sanitized[key] = value
    return sanitized


default_logger = setup_logging()
