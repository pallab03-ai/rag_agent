"""
RAG Pipeline for Document Loading and Processing

This module provides the complete RAG pipeline including document loading,
text splitting/chunking, embedding, and vector store creation.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)

from .utils import load_config, create_directory_if_not_exists
from .tools.vector_store import VectorStoreManager, initialize_vector_store


logger = logging.getLogger("rag_agent")


class DocumentProcessor:
    """
    Document processor for loading and chunking documents.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize DocumentProcessor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            separator: Separator for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[separator, "\n", " ", ""],
            length_function=len
        )
        
        logger.info(
            f"DocumentProcessor initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def load_documents_from_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to load (e.g., ['.txt', '.pdf'])
            
        Returns:
            List of Document objects
        """
        try:
            directory = Path(directory_path)
            
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if file_extensions is None:
                file_extensions = ['.txt', '.pdf', '.md', '.docx']
            
            logger.info(f"Loading documents from: {directory_path}")
            
            documents = []
            
            # Load each file type separately
            for ext in file_extensions:
                logger.info(f"Loading {ext} files...")
                docs = self._load_files_by_extension(directory, ext)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} {ext} files")
            
            logger.info(f"Total documents loaded: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from directory: {e}")
            raise
    
    def _load_files_by_extension(
        self,
        directory: Path,
        extension: str
    ) -> List[Document]:
        """
        Load files with specific extension from directory.
        
        Args:
            directory: Directory path
            extension: File extension (e.g., '.txt', '.pdf')
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Find all files with the extension
        files = list(directory.glob(f"**/*{extension}"))
        
        for file_path in files:
            try:
                logger.debug(f"Loading file: {file_path}")
                
                if extension == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                elif extension == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif extension == '.md':
                    loader = UnstructuredMarkdownLoader(str(file_path))
                else:
                    logger.warning(f"Unsupported file type: {extension}")
                    continue
                
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata['source'] = str(file_path)
                    doc.metadata['file_type'] = extension
                
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue
        
        return documents
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects
        """
        try:
            file = Path(file_path)
            
            if not file.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Loading document: {file_path}")
            
            extension = file.suffix.lower()
            
            if extension == '.txt':
                loader = TextLoader(str(file), encoding='utf-8')
            elif extension == '.pdf':
                loader = PyPDFLoader(str(file))
            elif extension == '.md':
                loader = UnstructuredMarkdownLoader(str(file))
            else:
                raise ValueError(f"Unsupported file type: {extension}")
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = str(file)
                doc.metadata['file_type'] = extension
            
            logger.info(f"Loaded {len(documents)} document(s)")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            if not documents:
                logger.warning("No documents to chunk")
                return []
            
            logger.info(f"Chunking {len(documents)} documents...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def process_documents(
        self,
        source_path: str,
        is_directory: bool = True
    ) -> List[Document]:
        """
        Complete document processing pipeline: load and chunk.
        
        Args:
            source_path: Path to document or directory
            is_directory: Whether source_path is a directory
            
        Returns:
            List of processed (chunked) Document objects
        """
        try:
            # Load documents
            if is_directory:
                documents = self.load_documents_from_directory(source_path)
            else:
                documents = self.load_single_document(source_path)
            
            if not documents:
                logger.warning("No documents loaded")
                return []
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            logger.info(f"Document processing complete: {len(chunks)} chunks ready")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise


class RAGPipeline:
    """
    Complete RAG pipeline combining document processing and vector store.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Configuration dictionary (optional)
        """
        if config is None:
            # Load default config
            try:
                config_path = Path(__file__).parent.parent / "configs" / "retrieval_config.yaml"
                config = load_config(str(config_path))
                self.config = config.get("retrieval", {})
            except Exception as e:
                logger.warning(f"Could not load config, using defaults: {e}")
                self.config = {}
        else:
            self.config = config
        
        # Initialize document processor
        chunk_size = self.config.get("chunk_size", 1000)
        chunk_overlap = self.config.get("chunk_overlap", 200)
        separator = self.config.get("separator", "\n\n")
        
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator
        )
        
        # Vector store manager (initialized later)
        self.vector_store_manager: Optional[VectorStoreManager] = None
        
        logger.info("RAG Pipeline initialized")
    
    def setup_knowledge_base(
        self,
        data_directory: str = "data",
        reset: bool = False,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None
    ) -> VectorStoreManager:
        """
        Setup complete knowledge base: load documents, chunk, embed, and store.
        
        Args:
            data_directory: Directory containing source documents
            reset: Whether to reset existing vector store
            embedding_provider: Embedding provider ("openai" or "nvidia")
            embedding_model: Embedding model name
            embedding_api_key: API key for embeddings
            embedding_base_url: Base URL for embeddings (for NVIDIA)
            embedding_base_url: Base URL for embeddings (for NVIDIA)
            
        Returns:
            VectorStoreManager instance
        """
        try:
            logger.info("=== Setting up Knowledge Base ===")
            
            # Process documents
            logger.info("Step 1: Processing documents...")
            chunks = self.document_processor.process_documents(
                source_path=data_directory,
                is_directory=True
            )
            
            if not chunks:
                raise ValueError("No documents were processed. Check data directory.")
            
            # Create vector store
            logger.info("Step 2: Creating vector store...")
            self.vector_store_manager = initialize_vector_store(
                documents=chunks,
                config=self.config,
                reset=reset,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_base_url=embedding_base_url
            )
            
            # Verify
            count = self.vector_store_manager.get_collection_count()
            logger.info(f"✓ Knowledge base setup complete with {count} document chunks")
            
            return self.vector_store_manager
            
        except Exception as e:
            logger.error(f"Error setting up knowledge base: {e}")
            raise
    
    def add_documents_to_kb(self, document_paths: List[str]) -> None:
        """
        Add new documents to existing knowledge base.
        
        Args:
            document_paths: List of document file paths
        """
        try:
            if self.vector_store_manager is None:
                raise ValueError("Knowledge base not initialized. Run setup_knowledge_base first.")
            
            logger.info(f"Adding {len(document_paths)} documents to knowledge base...")
            
            all_chunks = []
            
            for path in document_paths:
                chunks = self.document_processor.process_documents(
                    source_path=path,
                    is_directory=False
                )
                all_chunks.extend(chunks)
            
            if all_chunks:
                self.vector_store_manager.add_documents(all_chunks)
                logger.info(f"✓ Added {len(all_chunks)} new chunks to knowledge base")
            else:
                logger.warning("No new chunks to add")
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query_knowledge_base(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Query the knowledge base for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results (uses config default if None)
            
        Returns:
            List of relevant document contents
        """
        try:
            if self.vector_store_manager is None:
                raise ValueError("Knowledge base not initialized. Run setup_knowledge_base first.")
            
            if top_k is None:
                top_k = self.config.get("top_k", 5)
            
            logger.info(f"Querying knowledge base for: '{query[:50]}...'")
            
            results = self.vector_store_manager.similarity_search(
                query=query,
                k=top_k
            )
            
            documents = [doc.page_content for doc in results]
            
            logger.info(f"Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            raise
    
    def get_vector_store_manager(self) -> Optional[VectorStoreManager]:
        """
        Get the vector store manager instance.
        
        Returns:
            VectorStoreManager or None if not initialized
        """
        return self.vector_store_manager


def setup_rag_pipeline(
    data_directory: str = "data",
    config: Optional[Dict[str, Any]] = None,
    reset: bool = False
) -> RAGPipeline:
    """
    Convenience function to setup complete RAG pipeline.
    
    Args:
        data_directory: Directory containing documents
        config: Configuration dictionary (optional)
        reset: Whether to reset existing vector store
        
    Returns:
        Initialized RAGPipeline instance
    """
    pipeline = RAGPipeline(config=config)
    pipeline.setup_knowledge_base(data_directory=data_directory, reset=reset)
    return pipeline
