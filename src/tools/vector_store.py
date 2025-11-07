import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ..utils import get_env_variable, load_config
from ..embedding_factory import create_embeddings


logger = logging.getLogger("rag_agent")


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "rag_documents",
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.embeddings = self._initialize_embeddings()
        
        self.vectorstore: Optional[Chroma] = None
        
        logger.info(f"VectorStoreManager initialized with collection: {collection_name}")
    
    def _initialize_embeddings(self):
        embeddings = create_embeddings(
            provider=self.embedding_provider,
            model=self.embedding_model,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url
        )
        
        logger.info(f"Embeddings initialized with provider: {self.embedding_provider or 'default'}")
        return embeddings
    
    def create_vectorstore(
        self,
        documents: List[Document],
        reset: bool = False
    ) -> Chroma:
        try:
            if reset and Path(self.persist_directory).exists():
                logger.warning(f"Resetting vector store at {self.persist_directory}")
                # Delete existing collection by creating new vectorstore
            
            if not documents:
                raise ValueError("No documents provided to create vector store")
            
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            logger.info("Vector store created successfully")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vectorstore(self) -> Chroma:
        try:
            persist_path = Path(self.persist_directory)
            
            if not persist_path.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {self.persist_directory}. "
                    "Please create it first using create_vectorstore()."
                )
            
            logger.info(f"Loading vector store from {self.persist_directory}")
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            logger.info("Vector store loaded successfully")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        try:
            if self.vectorstore is None:
                logger.info("Vector store not loaded, loading now...")
                self.load_vectorstore()
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            ids = self.vectorstore.add_documents(documents)
            
            logger.info(f"Added {len(ids)} documents successfully")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            if self.vectorstore is None:
                logger.info("Vector store not loaded, loading now...")
                self.load_vectorstore()
            
            logger.info(f"Performing similarity search for: '{query[:50]}...'")
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        try:
            if self.vectorstore is None:
                logger.info("Vector store not loaded, loading now...")
                self.load_vectorstore()
            
            logger.info(f"Performing similarity search with scores for: '{query[:50]}...'")
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(f"Found {len(results)} relevant documents with scores")
            
            # Log scores for debugging
            for i, (doc, score) in enumerate(results, 1):
                logger.debug(f"Result {i}: Score = {score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {e}")
            raise
    
    def delete_collection(self) -> None:
        try:
            if self.vectorstore is not None:
                self.vectorstore.delete_collection()
                self.vectorstore = None
                logger.info(f"Deleted collection: {self.collection_name}")
            else:
                logger.warning("No vector store loaded to delete")
                
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_collection_count(self) -> int:
        try:
            if self.vectorstore is None:
                self.load_vectorstore()
            
            collection = self.vectorstore._collection
            count = collection.count()
            
            logger.info(f"Collection contains {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0


def initialize_vector_store(
    documents: List[Document],
    config: Optional[Dict[str, Any]] = None,
    reset: bool = False,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None
) -> VectorStoreManager:
    if config is None:
        # Load default config
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "retrieval_config.yaml"
            config = load_config(str(config_path))
            config = config.get("retrieval", {})
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            config = {}
    
    persist_directory = config.get("persist_directory", "chroma_db")
    collection_name = config.get("collection_name", "rag_documents")
    
    # Use provided embedding_model or fall back to config
    if embedding_model is None:
        embedding_model = config.get("embedding_model", None)
    
    manager = VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url
    )
    
    manager.create_vectorstore(documents, reset=reset)
    
    return manager


def retrieve_relevant_documents(
    query: str,
    vector_store_manager: VectorStoreManager,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None
) -> List[str]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: Search query
        vector_store_manager: VectorStoreManager instance
        top_k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score (optional)
        
    Returns:
        List of document contents as strings
    """
    try:
        if similarity_threshold:
            # Use search with scores for filtering
            results_with_scores = vector_store_manager.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # Filter by threshold
            filtered_results = [
                doc for doc, score in results_with_scores
                if score >= similarity_threshold
            ]
            
            documents = [doc.page_content for doc in filtered_results]
        else:
            # Regular similarity search
            results = vector_store_manager.similarity_search(
                query=query,
                k=top_k
            )
            documents = [doc.page_content for doc in results]
        
        logger.info(f"Retrieved {len(documents)} relevant documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []
