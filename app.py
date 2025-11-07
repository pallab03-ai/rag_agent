import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import RAGAgent
from src.rag_pipeline import RAGPipeline
from src.utils import setup_logging

load_dotenv()

st.set_page_config(
    page_title="RAG Agent - LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = setup_logging()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .local-source {
        background-color: #d4edda;
        color: #155724;
    }
    .web-source {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .none-source {
        background-color: #f8f9fa;
        color: #6c757d;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 RAG Agent with LangGraph</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Provider Selection
    st.subheader("🤖 LLM Provider")
    llm_provider = st.selectbox(
        "Select Provider",
        options=["OpenAI", "Groq", "NVIDIA"],
        index=0,
        help="Choose your LLM provider"
    )
    
    # API Key input based on provider
    st.subheader("🔑 API Keys")
    
    if llm_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value="",
            key="openai_llm_api_key",
            help="Enter your OpenAI API key"
        )
        
        model_name = st.text_input(
            "OpenAI Model Name",
            value="gpt-3.5-turbo",
            key="openai_llm_model",
            help="Enter the OpenAI model name (e.g., gpt-4, gpt-4-turbo, gpt-3.5-turbo)"
        )
        llm_base_url = None
        
    elif llm_provider == "Groq":
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value="",
            key="groq_api_key",
            help="Enter your Groq API key"
        )
        
        model_name = st.text_input(
            "Groq Model Name",
            value="llama-3.3-70b-versatile",
            key="groq_model",
            help="Enter the Groq model name (e.g., llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768)"
        )
        llm_base_url = None
        
    else:  # NVIDIA
        api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            value="",
            key="nvidia_llm_api_key",
            help="Enter your NVIDIA API key"
        )
        
        llm_base_url = st.text_input(
            "NVIDIA Base URL",
            value="https://integrate.api.nvidia.com/v1",
            key="nvidia_llm_base_url",
            help="NVIDIA API base URL (OpenAI-compatible endpoint)"
        )
        
        model_name = st.text_input(
            "NVIDIA Model Name",
            value="meta/llama-3.1-405b-instruct",
            key="nvidia_llm_model",
            help="Enter the NVIDIA model name (e.g., meta/llama-3.1-405b-instruct, nvidia/llama-3.1-nemotron-70b-instruct)"
        )
        st.info("ℹ️ NVIDIA uses OpenAI-compatible API")
    
    # Info message for embeddings
    if llm_provider in ["Groq", "NVIDIA"]:
        st.info("ℹ️ You can use OpenAI or NVIDIA for embeddings")
    
    # Embedding Provider Selection
    st.subheader("📊 Embedding Provider")
    embedding_provider = st.selectbox(
        "Select Embedding Provider",
        options=["OpenAI", "NVIDIA"],
        index=0,
        help="Choose your embedding provider for vector store"
    )
    
    if embedding_provider == "OpenAI":
        embedding_api_key = st.text_input(
            "OpenAI API Key (Embeddings)",
            type="password",
            value="",
            key="openai_embedding_api_key",
            help="Enter your OpenAI API key for embeddings"
        )
        embedding_model = st.text_input(
            "OpenAI Embedding Model Name",
            value="text-embedding-3-small",
            key="openai_embedding_model",
            help="Enter the OpenAI embedding model name (e.g., text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)"
        )
        embedding_base_url = None
    else:  # NVIDIA
        embedding_api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            value="",
            key="nvidia_embedding_api_key",
            help="Enter your NVIDIA API key for embeddings"
        )
        embedding_base_url = st.text_input(
            "NVIDIA Base URL",
            value="https://integrate.api.nvidia.com/v1",
            key="nvidia_embedding_base_url",
            help="NVIDIA API base URL (OpenAI-compatible endpoint)"
        )
        embedding_model = st.text_input(
            "NVIDIA Embedding Model",
            value="nvidia/embed-qa-4",
            key="nvidia_embedding_model",
            help="Enter the NVIDIA embedding model name. Try: nvidia/embed-qa-4 or NV-Embed-QA"
        )
        st.info("ℹ️ NVIDIA uses OpenAI-compatible API")
    
    tavily_key = st.text_input(
        "Tavily API Key (Optional)",
        type="password",
        key="tavily_api_key",
        value=os.getenv("TAVILY_API_KEY", ""),
        help="For enhanced web search capabilities"
    )
    
    # Agent settings
    st.subheader("⚙️ Agent Settings")
    enable_web_search = st.checkbox("Enable Web Search", value=True)
    enable_reflection = st.checkbox("Enable Reflection", value=True)
    show_trace = st.checkbox("Show Execution Trace", value=False)
    
    # LangSmith Tracing
    st.markdown("---")
    st.subheader("📊 LangSmith Tracing")
    enable_langsmith = st.checkbox(
        "Enable LangSmith Tracing",
        value=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        help="Track execution traces, performance, and costs in LangSmith"
    )
    
    if enable_langsmith:
        langsmith_key = st.text_input(
            "LangSmith API Key (Optional)",
            type="password",
            value=os.getenv("LANGCHAIN_API_KEY", ""),
            help="Get your API key from https://smith.langchain.com"
        )
        langsmith_project = st.text_input(
            "Project Name",
            value=os.getenv("LANGCHAIN_PROJECT", "rag-agent-langgraph"),
            help="Name for your LangSmith project"
        )
        
        if langsmith_key and langsmith_key != "your_langsmith_key_optional":
            st.info(f"🔗 View traces at: https://smith.langchain.com")
        else:
            st.warning("⚠️ Add LangSmith API key to enable tracing")
    
    st.markdown("---")
    
    # Document Upload Section
    st.subheader("📤 Upload Documents")
    st.markdown("Upload up to 3 documents (PDF, TXT, DOCX, or MD)")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx', 'md'],
        accept_multiple_files=True,
        help="Upload documents to add to your knowledge base",
        key="file_uploader"
    )
    
    # Validate file count
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning(f"⚠️ You can upload maximum 3 files. You selected {len(uploaded_files)} files.")
            uploaded_files = uploaded_files[:3]
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Show uploaded files
        for i, file in enumerate(uploaded_files, 1):
            st.text(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
        
        # Option to include existing data folder
        include_existing = st.checkbox(
            "Also include documents from 'data' folder",
            value=False,
            help="Check this to combine uploaded files with existing documents"
        )
    else:
        include_existing = False
    
    st.markdown("---")
    
    # Instructions
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Upload**: Upload up to 3 documents (optional)
    2. **Setup**: Click 'Initialize Knowledge Base'
    3. **Ask**: Enter your question
    4. **View**: See answer with source attribution
    5. **Explore**: Check execution trace for details
    """)
    
    st.markdown("---")
    
    # Knowledge Base Info
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        st.subheader("📚 Knowledge Base")
        try:
            doc_count = st.session_state.vector_store._collection.count()
            st.success(f"✅ {doc_count} documents loaded")
        except:
            st.info("Knowledge base active")

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Main content area
col1, col2 = st.columns([3, 1])

with col2:
    # Initialize button
    if st.button("🚀 Initialize Knowledge Base", type="primary", use_container_width=True):
        # Validate API keys based on provider
        if not api_key:
            st.error(f"❌ Please provide {llm_provider} API Key!")
        elif not embedding_api_key:
            st.error(f"❌ Please provide {embedding_provider} API Key for embeddings!")
        else:
            with st.spinner("🔄 Setting up knowledge base..."):
                try:
                    # Set LLM API keys and configuration
                    if llm_provider == "OpenAI":
                        os.environ["OPENAI_API_KEY"] = api_key
                        os.environ["LLM_PROVIDER"] = "openai"
                        os.environ["OPENAI_MODEL"] = model_name
                    elif llm_provider == "Groq":
                        os.environ["GROQ_API_KEY"] = api_key
                        os.environ["LLM_PROVIDER"] = "groq"
                        os.environ["GROQ_MODEL"] = model_name
                    else:  # NVIDIA
                        os.environ["NVIDIA_API_KEY"] = api_key
                        os.environ["LLM_PROVIDER"] = "nvidia"
                        os.environ["NVIDIA_MODEL"] = model_name
                        os.environ["NVIDIA_BASE_URL"] = llm_base_url
                    
                    # Set Embedding configuration
                    os.environ["EMBEDDING_PROVIDER"] = embedding_provider.lower()
                    if embedding_provider == "OpenAI":
                        os.environ["OPENAI_API_KEY"] = embedding_api_key
                    else:  # NVIDIA
                        os.environ["NVIDIA_API_KEY"] = embedding_api_key
                        os.environ["NVIDIA_BASE_URL"] = embedding_base_url
                    
                    if tavily_key:
                        os.environ["TAVILY_API_KEY"] = tavily_key
                    
                    # Setup LangSmith tracing
                    if enable_langsmith:
                        os.environ["LANGCHAIN_TRACING_V2"] = "true"
                        if langsmith_key and langsmith_key != "your_langsmith_key_optional":
                            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
                        if langsmith_project:
                            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
                        st.info("📊 LangSmith tracing enabled")
                    else:
                        os.environ["LANGCHAIN_TRACING_V2"] = "false"
                    
                    # Handle uploaded files
                    data_dir = Path(__file__).parent / "data"
                    upload_dir = Path(__file__).parent / "uploaded_docs"
                    chroma_dir = Path(__file__).parent / "chroma_db"
                    upload_dir.mkdir(exist_ok=True)
                    
                    # IMPORTANT: Clear ChromaDB to ensure no old data remains
                    if chroma_dir.exists():
                        import shutil
                        st.info("🧹 Clearing old vector database...")
                        shutil.rmtree(chroma_dir)
                        logger.info("Cleared ChromaDB directory")
                    
                    # Clean up old uploaded files
                    for old_file in upload_dir.glob("*"):
                        if old_file.is_file():
                            old_file.unlink()
                    
                    # Save uploaded files if any
                    if uploaded_files:
                        st.info(f"📥 Processing {len(uploaded_files)} uploaded file(s)...")
                        for uploaded_file in uploaded_files:
                            file_path = upload_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            logger.info(f"Saved uploaded file: {uploaded_file.name}")
                        
                        if include_existing and data_dir.exists():
                            # Combine both directories
                            st.info("📚 Combining uploaded files with existing documents...")
                            # We'll load from both directories by copying files
                            import shutil
                            for existing_file in data_dir.glob("*"):
                                if existing_file.is_file():
                                    shutil.copy(existing_file, upload_dir / existing_file.name)
                            documents_dir = str(upload_dir)
                        else:
                            # Use only uploaded_docs directory
                            documents_dir = str(upload_dir)
                    else:
                        # Use default data directory
                        documents_dir = str(data_dir)
                    
                    # Initialize RAG Pipeline
                    rag_pipeline = RAGPipeline()
                    
                    # Setup knowledge base with embedding configuration
                    # This will load documents, chunk them, and create vector store
                    vector_store_manager = rag_pipeline.setup_knowledge_base(
                        data_directory=documents_dir,
                        reset=True,  # Reset to avoid duplicates
                        embedding_provider=embedding_provider.lower(),
                        embedding_model=embedding_model,
                        embedding_api_key=embedding_api_key,
                        embedding_base_url=embedding_base_url
                    )
                    
                    st.session_state.vector_store = vector_store_manager.vectorstore
                    logger.info("Vector store created successfully")
                    
                    # Initialize agent
                    agent = RAGAgent(
                        vector_store_manager=vector_store_manager,
                        enable_reflection=enable_reflection
                    )
                    
                    st.session_state.agent = agent
                    st.session_state.initialized = True
                    
                    st.success("✅ Knowledge base initialized!")
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Initialization error: {e}")
                    st.error(f"❌ Initialization failed: {str(e)}")

with col1:
    # Question input
    st.subheader("💬 Ask a Question")
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the main types of renewable energy?",
        help="Ask about renewable energy, climate change, or sustainability"
    )
    
    # Submit button
    col_submit, col_clear = st.columns([4, 1])
    with col_submit:
        submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# Process question
if submit_button:
    if not st.session_state.initialized:
        st.error("❌ Please initialize the knowledge base first!")
    elif not question.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Run agent
                result = st.session_state.agent.run(question)
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 Results")
    
    # Show most recent first
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"❓ {item['question']}", expanded=(idx == 0)):
            result = item['result']
            
            # Answer
            st.markdown("### 🎯 Answer")
            
            # Source badge - Show prominently at the top
            source = result.get('source', 'none')
            if source == 'local':
                badge_class = 'local-source'
                badge_icon = '📚'
                badge_text = 'Vector Database (Local Knowledge Base)'
            elif source == 'web':
                badge_class = 'web-source'
                badge_icon = '🌐'
                badge_text = 'Web Search (Internet)'
            else:
                badge_class = 'none-source'
                badge_icon = '💬'
                badge_text = 'Direct Answer (No External Source)'
            
            st.markdown(
                f'<div class="source-badge {badge_class}">{badge_icon} Source: {badge_text}</div>',
                unsafe_allow_html=True
            )
            
            st.write(result.get('answer', 'No answer available'))
            
            # Show retrieval details
            if source == 'local' and result.get('retrieved_docs'):
                st.info(f"ℹ️ Retrieved {len(result['retrieved_docs'])} documents from vector database")
            elif source == 'web' and result.get('web_search_results'):
                st.info(f"ℹ️ Found {len(result['web_search_results'])} results from web search")
            
            # Reflection metrics
            if enable_reflection and result.get('reflection'):
                st.markdown("### 🤔 Reflection")
                reflection = result['reflection']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Relevant", "✅ Yes" if reflection.get('is_relevant') else "❌ No")
                with col2:
                    st.metric("Complete", "✅ Yes" if reflection.get('is_complete') else "❌ No")
                with col3:
                    confidence = reflection.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.2%}")
                
                if reflection.get('strengths'):
                    st.markdown(f"**✨ Strengths:** {reflection['strengths']}")
                if reflection.get('weaknesses'):
                    st.markdown(f"**⚠️ Weaknesses:** {reflection['weaknesses']}")
                if reflection.get('missing_information'):
                    st.markdown(f"**❓ Missing Info:** {reflection['missing_information']}")
            
            # Execution trace
            if show_trace:
                st.markdown("### 🔍 Execution Trace")
                
                with st.expander("View Details"):
                    # Plan
                    if result.get('query_plan'):
                        st.markdown("**📝 Query Plan:**")
                        plan = result['query_plan']
                        st.json({
                            'needs_retrieval': plan.get('needs_retrieval'),
                            'question_type': plan.get('question_type'),
                            'search_terms': plan.get('search_terms', [])
                        })
                    
                    # Retrieved documents
                    if result.get('retrieved_docs'):
                        st.markdown(f"**📄 Retrieved Documents:** {len(result['retrieved_docs'])}")
                        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                            # doc is already a string, not a Document object
                            doc_text = doc if isinstance(doc, str) else doc.page_content
                            st.text(f"{i}. {doc_text[:200]}...")
                    
                    # Web search results
                    if result.get('web_search_results'):
                        st.markdown(f"**🔎 Web Results:** {len(result['web_search_results'])}")
                        for i, res in enumerate(result['web_search_results'][:3], 1):
                            # res is a formatted string, not a dict
                            st.text(f"{i}. {res[:150]}...")
                    
                    # Context
                    if result.get('context'):
                        st.markdown("**📝 Context Used:**")
                        st.text(result['context'][:500] + "...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 1rem;'>
    <p>Built with LangGraph 🦜🔗 | Powered by ChromaDB 🗄️ | Enhanced by Tavily 🔎</p>
    <p><small>Topics: Renewable Energy • Climate Change • Sustainability</small></p>
</div>
""", unsafe_allow_html=True)
