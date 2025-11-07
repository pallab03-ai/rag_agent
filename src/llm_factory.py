import os
import logging
from typing import Optional

logger = logging.getLogger("rag_agent")


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
):
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    else:
        provider = provider.lower()
    
    logger.info(f"Creating LLM with provider: {provider}")
    
    if provider == "openai":
        return _create_openai_llm(model, api_key, temperature, max_tokens, **kwargs)
    elif provider == "groq":
        return _create_groq_llm(model, api_key, temperature, max_tokens, **kwargs)
    elif provider == "nvidia":
        return _create_nvidia_llm(model, api_key, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai', 'groq', or 'nvidia'")


def _create_openai_llm(model, api_key, temperature, max_tokens, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("langchain-openai is required for OpenAI provider. Install with: pip install langchain-openai")
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
    
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    logger.info(f"Creating OpenAI LLM with model: {model}")
    
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def _create_groq_llm(model, api_key, temperature, max_tokens, **kwargs):
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("langchain-groq is required for Groq provider. Install with: pip install langchain-groq")
    
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter")
    
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    logger.info(f"Creating Groq LLM with model: {model}")
    
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def _create_nvidia_llm(model, api_key, temperature, max_tokens, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("langchain-openai is required for NVIDIA provider. Install with: pip install langchain-openai")
    
    if api_key is None:
        api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError("NVIDIA API key is required. Set NVIDIA_API_KEY environment variable or pass api_key parameter")
    
    base_url = kwargs.pop("base_url", None) or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    if model is None:
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-405b-instruct")
    
    logger.info(f"Creating NVIDIA LLM with model: {model}, base_url: {base_url}")
    
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def get_available_providers():
    providers = []
    
    try:
        import langchain_openai
        providers.append("openai")
        providers.append("nvidia")
    except ImportError:
        pass
    
    try:
        import langchain_groq
        providers.append("groq")
    except ImportError:
        pass
    
    return providers


def get_nvidia_models():
    return [
        ("meta/llama-3.1-405b-instruct", "Llama 3.1 405B Instruct"),
        ("nvidia/llama-3.1-nemotron-70b-instruct", "Nemotron 70B Instruct"),
        ("meta/llama-3.1-70b-instruct", "Llama 3.1 70B Instruct"),
        ("meta/llama-3.1-8b-instruct", "Llama 3.1 8B Instruct"),
    ]


def get_groq_models():
    return [
        ("llama-3.3-70b-versatile", "Llama 3.3 70B (Versatile)"),
        ("llama-3.1-70b-versatile", "Llama 3.1 70B (Versatile)"),
        ("llama-3.1-8b-instant", "Llama 3.1 8B (Instant)"),
        ("mixtral-8x7b-32768", "Mixtral 8x7B"),
        ("gemma2-9b-it", "Gemma 2 9B"),
    ]


def get_openai_models():
    return [
        ("gpt-4", "GPT-4"),
        ("gpt-4-turbo", "GPT-4 Turbo"),
        ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ]
