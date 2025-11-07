import os
import logging
from typing import Optional, List

logger = logging.getLogger("rag_agent")


class NVIDIAEmbeddingsWrapper:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "input": text,
                "input_type": "passage"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embeddings.append(result["data"][0]["embedding"])
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        import requests
        
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": text,
            "input_type": "query"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if "data" in result and len(result["data"]) > 0:
            return result["data"][0]["embedding"]
        else:
            raise ValueError(f"Unexpected response format: {result}")


def create_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
):
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    else:
        provider = provider.lower()
    
    logger.info(f"Creating embeddings with provider: {provider}")
    
    if provider == "openai":
        return _create_openai_embeddings(model, api_key, **kwargs)
    elif provider == "nvidia":
        return _create_nvidia_embeddings(model, api_key, base_url, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Use 'openai' or 'nvidia'")


def _create_openai_embeddings(model, api_key, **kwargs):
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError("langchain-openai is required for OpenAI embeddings. Install with: pip install langchain-openai")
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
    
    if model is None:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    logger.info(f"Creating OpenAI embeddings with model: {model}")
    
    return OpenAIEmbeddings(
        api_key=api_key,
        model=model,
        **kwargs
    )


def _create_nvidia_embeddings(model, api_key, base_url, **kwargs):
    if api_key is None:
        api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError(
            "NVIDIA API key is required. "
            "Get your API key from: https://build.nvidia.com/explore/discover"
        )
    
    if model is None:
        model = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/embed-qa-4")
    
    if base_url is None:
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    logger.info(f"Creating NVIDIA embeddings with model: {model}")
    logger.info(f"Using NVIDIA base URL: {base_url}")
    
    try:
        embeddings = NVIDIAEmbeddingsWrapper(
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"✓ NVIDIA embeddings initialized with model: {model}")
        
        try:
            logger.info("Testing NVIDIA embeddings with sample text...")
            test_result = embeddings.embed_query("test")
            logger.info(f"✓ NVIDIA embeddings test successful (dimension: {len(test_result)})")
        except Exception as test_error:
            error_str = str(test_error)
            if "404" in error_str:
                raise ValueError(
                    f"NVIDIA model '{model}' returned 404 error.\n\n"
                    f"This usually means:\n"
                    f"1. Your API key doesn't have access to this model\n"
                    f"2. The model name is incorrect\n"
                    f"3. You need to enable the model at https://build.nvidia.com\n\n"
                    f"Please check:\n"
                    f"- Visit https://build.nvidia.com/{model.replace('/', '/')}\n"
                    f"- Ensure you have access to the model\n"
                    f"- Try using 'NV-Embed-QA' as an alternative\n\n"
                    f"Original error: {error_str}"
                )
            raise
        
        return embeddings
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error creating NVIDIA embeddings: {error_msg}")
        raise ValueError(
            f"Failed to initialize NVIDIA embeddings: {error_msg}\n"
            f"Make sure your API key is valid and has access to the model.\n"
            f"Visit https://build.nvidia.com/explore/discover to manage access."
        )


def get_available_embedding_providers():
    providers = []
    
    try:
        import langchain_openai
        providers.append("openai")
    except ImportError:
        pass
    
    try:
        import langchain_nvidia_ai_endpoints
        providers.append("nvidia")
    except ImportError:
        pass
    
    return providers


def get_nvidia_embedding_models():
    return [
        ("nvidia/llama-3.2-nemoretriever-300m-embed-v2", "Llama 3.2 NeMo Retriever 300M Embed v2"),
        ("nvidia/nv-embedqa-e5-v5", "NV-EmbedQA E5 v5"),
        ("nvidia/nv-embed-v2", "NV-Embed v2"),
    ]


def get_openai_embedding_models():
    return [
        ("text-embedding-3-small", "Text Embedding 3 Small (1536 dims)"),
        ("text-embedding-3-large", "Text Embedding 3 Large (3072 dims)"),
        ("text-embedding-ada-002", "Text Embedding Ada 002 (Legacy)"),
    ]
