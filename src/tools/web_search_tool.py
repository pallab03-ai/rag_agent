import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils import get_env_variable, load_config, truncate_text


logger = logging.getLogger("rag_agent")


class WebSearchTool:
    def __init__(
        self,
        provider: str = "tavily",
        fallback_provider: str = "duckduckgo",
        max_results: int = 5,
        timeout: int = 10
    ):
        self.provider = provider.lower()
        self.fallback_provider = fallback_provider.lower()
        self.max_results = max_results
        self.timeout = timeout
        
        self.tavily_client = None
        self.serpapi_client = None
        
        self._initialize_providers()
        
        logger.info(f"WebSearchTool initialized with provider: {self.provider}")
    
    def _initialize_providers(self) -> None:
        # Initialize Tavily
        if self.provider == "tavily" or self.fallback_provider == "tavily":
            try:
                from tavily import TavilyClient
                api_key = get_env_variable("TAVILY_API_KEY")
                if api_key:
                    self.tavily_client = TavilyClient(api_key=api_key)
                    logger.info("Tavily client initialized")
                else:
                    logger.warning("TAVILY_API_KEY not found")
            except ImportError:
                logger.warning("Tavily package not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily: {e}")
        
        # Initialize SerpAPI
        if self.provider == "serpapi" or self.fallback_provider == "serpapi":
            try:
                api_key = get_env_variable("SERPAPI_API_KEY")
                if api_key:
                    self.serpapi_client = api_key  # Store key for later use
                    logger.info("SerpAPI configured")
                else:
                    logger.warning("SERPAPI_API_KEY not found")
            except Exception as e:
                logger.warning(f"Failed to configure SerpAPI: {e}")
    
    def search(self, query: str, use_fallback: bool = True) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Performing web search with {self.provider} for: '{query[:50]}...'")
            
            results = self._search_with_provider(query, self.provider)
            
            if results:
                logger.info(f"Found {len(results)} results with {self.provider}")
                return results
            
            if use_fallback and self.fallback_provider != self.provider:
                logger.warning(f"Primary search failed, trying fallback: {self.fallback_provider}")
                results = self._search_with_provider(query, self.fallback_provider)
                
                if results:
                    logger.info(f"Found {len(results)} results with fallback {self.fallback_provider}")
                    return results
            
            logger.warning("No search results found")
            return []
            
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return []
    
    def _search_with_provider(self, query: str, provider: str) -> List[Dict[str, Any]]:
        if provider == "tavily":
            return self._search_tavily(query)
        elif provider == "duckduckgo":
            return self._search_duckduckgo(query)
        elif provider == "serpapi":
            return self._search_serpapi(query)
        else:
            logger.error(f"Unknown provider: {provider}")
            return []
    
    def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            if not self.tavily_client:
                logger.warning("Tavily client not initialized")
                return []
            
            # Perform search
            response = self.tavily_client.search(
                query=query,
                max_results=self.max_results,
                search_depth="advanced"
            )
            
            # Format results
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []
    
    def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo (free, no API key required).
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            from duckduckgo_search import DDGS
            
            results = []
            
            # Perform search
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    keywords=query,
                    max_results=self.max_results,
                    region="wt-wt",  # Worldwide
                    safesearch="moderate"
                )
                
                for item in search_results:
                    results.append({
                        "title": item.get("title", "No title"),
                        "url": item.get("href", "") or item.get("link", ""),
                        "content": item.get("body", "") or item.get("snippet", "")
                    })
            
            return results
            
        except ImportError:
            logger.error("duckduckgo-search package not installed")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_serpapi(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using SerpAPI.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            if not self.serpapi_client:
                logger.warning("SerpAPI key not configured")
                return []
            
            from serpapi import GoogleSearch
            
            params = {
                "q": query,
                "api_key": self.serpapi_client,
                "num": self.max_results,
                "engine": "google"
            }
            
            search = GoogleSearch(params)
            response = search.get_dict()
            
            # Format results
            results = []
            for item in response.get("organic_results", [])[:self.max_results]:
                results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("link", ""),
                    "content": item.get("snippet", "")
                })
            
            return results
            
        except ImportError:
            logger.error("google-search-results package not installed")
            return []
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a readable string.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted results string
        """
        if not results:
            return "No search results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content available")
            
            # Truncate content if too long
            content = truncate_text(content, max_length=300)
            
            formatted.append(
                f"Result {i}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
            )
        
        return "\n".join(formatted)


def create_web_search_tool(config: Optional[Dict[str, Any]] = None) -> WebSearchTool:
    """
    Create WebSearchTool with configuration.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        WebSearchTool instance
    """
    if config is None:
        # Load default config
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "web_search_config.yaml"
            config = load_config(str(config_path))
            config = config.get("web_search", {})
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            config = {}
    
    provider = config.get("provider", "tavily")
    fallback_provider = config.get("fallback_provider", "duckduckgo")
    max_results = config.get("max_results", 5)
    timeout = config.get("timeout", 10)
    
    tool = WebSearchTool(
        provider=provider,
        fallback_provider=fallback_provider,
        max_results=max_results,
        timeout=timeout
    )
    
    return tool


def perform_web_search(
    query: str,
    max_results: int = 5,
    provider: str = "tavily",
    fallback: bool = True
) -> List[Dict[str, Any]]:
    """
    Convenience function to perform web search.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        provider: Primary search provider
        fallback: Whether to use fallback on failure
        
    Returns:
        List of search result dictionaries
    """
    try:
        tool = WebSearchTool(
            provider=provider,
            max_results=max_results
        )
        
        results = tool.search(query, use_fallback=fallback)
        return results
        
    except Exception as e:
        logger.error(f"Error in perform_web_search: {e}")
        return []
