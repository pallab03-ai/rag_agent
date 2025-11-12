# RAG Agent with LangGraph 🤖

A production-ready Retrieval-Augmented Generation (RAG) AI Agent built with **LangGraph** that intelligently answers questions using local knowledge base with web search fallback.

> 📖 **New:** Read about the [AI/ML Project Experience](PROJECT_EXPERIENCE.md) - challenges, solutions, and key takeaways from building this RAG system.

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.9+
- API keys for at least one provider combination:
  - **OpenAI** (for both LLM + Embeddings), OR
  - **Groq** (LLM) + **NVIDIA** (Embeddings), OR
  - **Groq** (LLM) + **OpenAI** (Embeddings)

### Installation

1. **Clone and setup environment**

```bash
git clone <repository-url>
cd New_rag_agent
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

2. **Configure API Keys**

Create a `.env` file in the root directory and add your API keys:

**Option 1: OpenAI Only**

```env
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Embedding Provider
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional: Web Search
TAVILY_API_KEY=your-tavily-key-here
ENABLE_WEB_SEARCH=true
```

**Option 2: Groq + NVIDIA (Cost-effective)**

```env
# LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile

# Embedding Provider
EMBEDDING_PROVIDER=nvidia
NVIDIA_API_KEY=your-nvidia-api-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_EMBEDDING_MODEL=nvidia/llama-3.2-nemoretriever-300m-embed-v1

# Optional: Web Search
TAVILY_API_KEY=your-tavily-key-here
ENABLE_WEB_SEARCH=true
```

**Option 3: Mix and Match**

```env
# Use Groq for fast LLM inference
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile

# Use OpenAI for embeddings
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional: Web Search
TAVILY_API_KEY=your-tavily-key-here
ENABLE_WEB_SEARCH=true
```

### 🎯 Get API Keys

#### OpenAI

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

**Recommended Models:**

- LLM: `gpt-3.5-turbo` (fast, cost-effective) or `gpt-4` (higher quality)
- Embeddings: `text-embedding-3-small` (cost-effective) or `text-embedding-3-large` (higher quality)

#### Groq

1. Go to https://console.groq.com/keys
2. Create account and generate API key
3. Copy the key (starts with `gsk_`)
4. Add to `.env`: `GROQ_API_KEY=gsk_...`

**Recommended Models:**

- `llama-3.3-70b-versatile` (balanced)
- `llama-3.1-70b-versatile` (fast)
- `mixtral-8x7b-32768` (large context)

#### NVIDIA

1. Go to https://build.nvidia.com/
2. Create account and get API key
3. Copy the key (starts with `nvapi-`)
4. Add to `.env`: `NVIDIA_API_KEY=nvapi-...`

**Recommended Models:**

- `nvidia/llama-3.2-nemoretriever-300m-embed-v1` (2048 dim)
- `nvidia/nv-embedqa-e5-v5` (1024 dim)

#### Tavily (Optional - for web search)

1. Go to https://tavily.com/
2. Sign up for free account
3. Copy API key
4. Add to `.env`: `TAVILY_API_KEY=tvly-...`

### 3. Run the Application

**Launch Streamlit UI (Easiest)**

```bash
streamlit run app.py
```

Then:

1. Select your **LLM Provider** (OpenAI/Groq/NVIDIA)
2. Enter your **API key** and **model name**
3. Select your **Embedding Provider** (OpenAI/NVIDIA)
4. Enter embedding **API key** and **model name**
5. (Optional) Enter **Tavily API key** for web search
6. Click **"Initialize Knowledge Base"**
7. Upload documents or use sample data
8. Start asking questions!

### 📊 Example Usage

**Sample Questions to Try:**

Local Knowledge Base:

- "What are the main types of renewable energy?"
- "Explain the greenhouse effect."
- "What are the three pillars of sustainability?"

Web Search (if enabled):

- "What are the latest renewable energy breakthroughs?"
- "Current global solar energy capacity"

General Questions:

- "Hello, how are you?"
- "What is 2 + 2?"

### 🔧 Python API Usage

```python
from src.agent import create_rag_agent
from src.tools.vector_store import VectorStoreManager

# Initialize vector store with your chosen embedding provider
vector_store = VectorStoreManager(
    persist_directory="chroma_db",
    collection_name="rag_documents",
    embedding_provider="openai",  # or "nvidia"
    embedding_model="text-embedding-3-small",
    embedding_api_key="your-api-key"
)

# Load existing vector store
vector_store.load_vectorstore()

# Create RAG agent
agent = create_rag_agent(
    vector_store_manager=vector_store,
    enable_reflection=True,
    log_level="INFO"
)

# Ask questions
result = agent.run("What is renewable energy?")

print(f"Answer: {result['answer']}")
print(f"Source: {result['source']}")
if result.get('reflection'):
    print(f"Confidence: {result['reflection']['confidence']}")
```

## 🌟 Key Features

- **🧠 Intelligent Query Planning** - Analyzes questions to determine retrieval strategy
- **📚 Local Knowledge Base** - ChromaDB vector store with flexible embedding providers
- **🌐 Web Search Fallback** - Automatically searches web when local info insufficient
- **� Relevance Checking** - LLM-based evaluation of retrieved documents
- **🤔 Answer Reflection** - Self-evaluation of answer quality
- **🎯 Multi-Source Attribution** - Clear indication of answer sources
- **🤖 Multiple LLM Providers** - OpenAI, Groq, NVIDIA support
- **� Multiple Embedding Providers** - OpenAI, NVIDIA support
- **💰 Cost Optimization** - Mix and match providers
- **⚡ High Performance** - Fast inference with provider flexibility

## 🏗️ Architecture

The agent uses a **LangGraph StateGraph** with 6 specialized nodes:

```
START → PLAN → (conditional) → RETRIEVE → CHECK_RELEVANCE → (conditional) → WEB_SEARCH → ANSWER → REFLECT → END
```

### Workflow Nodes

1. **PLAN** - Analyzes the question and determines retrieval needs
2. **RETRIEVE** - Queries the ChromaDB vector store for relevant documents
3. **CHECK_RELEVANCE** - Evaluates if retrieved documents are relevant
4. **WEB_SEARCH** - Falls back to web search if needed (Tavily/DuckDuckGo/SerpAPI)
5. **ANSWER** - Generates the final answer using GPT-4 or GPT-3.5-turbo
6. **REFLECT** - Evaluates answer quality, completeness, and confidence

### Conditional Routing

- After **PLAN**: Routes to RETRIEVE if retrieval needed, otherwise directly to ANSWER
- After **CHECK_RELEVANCE**: Routes to WEB_SEARCH if documents not relevant, otherwise to ANSWER

## 📦 Tech Stack

### Core Framework

- **LangGraph** (0.0.40+) - Agent orchestration with StateGraph
- **LangChain** (0.1.0+) - RAG pipeline and LLM integration
- **ChromaDB** (0.4.22+) - Local vector database with persistence

### LLM Providers

- **OpenAI** - GPT-4, GPT-4-turbo, GPT-3.5-turbo for generation
- **Groq** - llama-3.3-70b, llama-3.1-70b, mixtral-8x7b for fast inference

### Embedding Providers

- **OpenAI** - text-embedding-3-small, text-embedding-3-large
- **NVIDIA** - llama-3.2-nemoretriever-300m-embed-v2, nv-embedqa-e5-v5, nv-embed-v2

### Additional Tools

- **Tavily/DuckDuckGo/SerpAPI** - Web search providers with automatic fallback
- **Streamlit** (1.29.0+) - Interactive web UI
- **RAGAS** (0.1.0+) - Evaluation metrics
- **pypdf, python-docx** - Document processing

## 🚀 Installation

### Prerequisites

- Python 3.9+
- **At least one LLM provider API key:**
  - OpenAI API Key, OR
  - Groq API Key
- **At least one embedding provider API key:**
  - OpenAI API Key, OR
  - NVIDIA API Key + Base URL
- Tavily API Key (optional, for enhanced web search)

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd New_rag_agent
```

2. **Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
notepad .env  # Windows
nano .env     # Linux/Mac
```

Configure based on your provider choice:

**Option 1: Groq + NVIDIA (Recommended for cost optimization)**

```env
# LLM Configuration
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Embedding Configuration
EMBEDDING_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_EMBEDDING_MODEL=nvidia/llama-3.2-nemoretriever-300m-embed-v2

# Optional: Web Search
TAVILY_API_KEY=your_tavily_api_key_here
```

**Option 2: OpenAI for both (Simplest setup)**

```env
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Embedding Configuration
EMBEDDING_PROVIDER=openai
# Same OPENAI_API_KEY used for embeddings

# Optional: Web Search
TAVILY_API_KEY=your_tavily_api_key_here
```

See [PROVIDER_SETUP_GUIDE.md](PROVIDER_SETUP_GUIDE.md) for more configuration options.

## 📖 Usage

### 1. Streamlit Web UI (Recommended)

The easiest way to interact with the agent:

```bash
streamlit run app.py
```

Then:

1. **Select LLM Provider** (OpenAI or Groq)
2. Enter LLM API key and choose model
3. **Select Embedding Provider** (OpenAI or NVIDIA)
4. Enter embedding API key (and base URL for NVIDIA)
5. Optional: Enter Tavily API key for web search
6. Click "Initialize Knowledge Base"
7. Start asking questions!

Features:

- ✨ Beautiful, intuitive interface
- 🔧 Dynamic provider selection
- 📊 Real-time answer generation
- 🔍 Source attribution (Local KB / Web Search / Direct)
- 🤔 Reflection metrics (relevance, completeness, confidence)
- 📝 Execution trace viewer
- 💾 Conversation history

### 3. Evaluation

Run comprehensive evaluation with multiple metrics:

```bash
python evaluate_agent.py
```

This will:

- Test the agent on diverse questions
- Run **Basic Evaluation** (LLM-as-Judge, response time, source accuracy)
- Run **Comprehensive Evaluation** (RAGAs, BLEU, ROUGE, BERTScore) - if packages installed
- Calculate success rates by category
- Generate detailed reports in `evaluation_results/`

#### Evaluation Modes

**Basic Evaluation** (always available):

- LLM-as-Judge metrics (answer quality, relevance)
- Response time measurement
- Source accuracy checking

**Comprehensive Evaluation** (optional, requires additional packages):

- RAGAs metrics: faithfulness, answer_relevancy, context_relevancy, context_recall, context_precision
- NLP metrics: BLEU-1/2/4, ROUGE-1/2/L, BERTScore
- Custom metrics: answer length, context usage rate

To enable comprehensive evaluation:

```bash
pip install -r requirements-eval.txt
```

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed evaluation documentation.

## 🗂️ Project Structure

```
New_rag_agent/
├── src/                       # Core application code
│   ├── __init__.py
│   ├── agent.py              # Main RAGAgent class with LangGraph workflow
│   ├── llm_factory.py        # LLM provider factory (OpenAI/Groq/NVIDIA)
│   ├── embedding_factory.py  # Embedding provider factory (OpenAI/NVIDIA)
│   ├── state.py              # AgentState TypedDict definition
│   ├── routing.py            # Conditional routing functions
│   ├── utils.py              # Utility functions (logging, parsing, formatting)
│   ├── rag_pipeline.py       # Document loading and processing pipeline
│   ├── langsmith_config.py   # LangSmith tracing configuration
│   ├── evaluation_metrics.py # Evaluation metrics (RAGAs, BLEU, ROUGE, BERTScore)
│   ├── nodes/                # LangGraph workflow nodes
│   │   ├── __init__.py
│   │   ├── plan.py           # Query planning and analysis
│   │   ├── retrieve.py       # Vector database retrieval
│   │   ├── check_relevance.py # Document relevance checking
│   │   ├── web_search.py     # Web search fallback
│   │   ├── answer.py         # Answer generation
│   │   └── reflect.py        # Answer quality reflection
│   └── tools/                # External tool integrations
│       ├── __init__.py
│       ├── vector_store.py   # ChromaDB vector store manager
│       └── web_search_tool.py # Multi-provider web search (Tavily/DuckDuckGo)
├── data/                      # Sample documents for knowledge base
│   ├── renewable_energy.txt
│   ├── climate_change.txt
│   └── sustainability.txt
├── chroma_db/                 # ChromaDB vector store persistence
├── uploaded_docs/             # User-uploaded documents via Streamlit
├── logs/                      # Application logs
├── evaluation_results/        # Evaluation metrics and reports
├── app.py                     # Streamlit web UI application
├── evaluate_agent.py          # Agent evaluation runner
├── requirements.txt           # Python dependencies
├── requirements-eval.txt      # Optional evaluation dependencies
├── .env                       # Environment variables (API keys, config)
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore patterns
├── README.md                 # This file - project documentation
├── PROJECT_EXPERIENCE.md     # AI/ML project experience and learnings
├── ARCHITECTURE.md           # System architecture details
├── EVALUATION_GUIDE.md       # Evaluation metrics documentation
├── LANGSMITH_GUIDE.md        # LangSmith tracing setup guide
└── PROVIDER_SETUP_GUIDE.md   # Multi-provider configuration guide
```

### Key Files Explained

**Core Agent Files:**

- `src/agent.py` - Main RAGAgent class implementing LangGraph StateGraph
- `src/llm_factory.py` - Factory pattern for creating LLM instances (supports OpenAI, Groq, NVIDIA)
- `src/embedding_factory.py` - Factory pattern for creating embedding instances (supports OpenAI, NVIDIA)
- `src/state.py` - AgentState TypedDict that maintains workflow state across nodes

**Workflow Nodes:**

- `src/nodes/plan.py` - Analyzes questions and determines if retrieval is needed
- `src/nodes/retrieve.py` - Queries ChromaDB vector store for relevant documents
- `src/nodes/check_relevance.py` - Evaluates if retrieved documents answer the question
- `src/nodes/web_search.py` - Performs web search when local documents are insufficient
- `src/nodes/answer.py` - Generates final answer using LLM with context
- `src/nodes/reflect.py` - Evaluates answer quality and completeness

**Tools & Utilities:**

- `src/tools/vector_store.py` - ChromaDB vector store manager with CRUD operations
- `src/tools/web_search_tool.py` - Multi-provider web search with fallback support
- `src/utils.py` - Logging, formatting, parsing helper functions
- `src/routing.py` - Conditional routing logic for LangGraph edges

**User Interfaces:**

- `app.py` - Streamlit web application with provider selection UI
- `evaluate_agent.py` - Evaluation script with comprehensive metrics

**Data & Storage:**

- `data/` - Sample documents for testing
- `chroma_db/` - Persistent vector database storage
- `uploaded_docs/` - User-uploaded documents via Streamlit UI
- `logs/` - Application execution logs
- `evaluation_results/` - Evaluation reports and metrics

## 📊 Sample Questions

### Local Knowledge Base Questions (will use ChromaDB)

- "What are the main types of renewable energy?"
- "How does solar panel efficiency compare to wind turbines?"
- "What are the three pillars of sustainability?"
- "Explain the greenhouse effect and its role in global warming."

### Web Search Questions (will search the web)

- "What are the latest fusion energy breakthroughs in 2024?"
- "What is the current global renewable energy capacity?"
- "Which country recently announced the largest solar farm project?"

### Simple Questions (no retrieval needed)

- "Hello, how are you?"
- "What is 2 + 2?"
- "Thank you for your help!"

## 🔍 How It Works

### 1. Question Analysis (PLAN Node)

The agent first analyzes the question using an LLM to determine:

- Does this question need information retrieval?
- What search terms should be used?
- What type of question is this?

### 2. Retrieval (RETRIEVE Node)

If retrieval is needed, the agent:

- Queries the ChromaDB vector store
- Retrieves top-k most similar documents
- Formats them as context

### 3. Relevance Check (CHECK_RELEVANCE Node)

An LLM evaluates whether the retrieved documents contain relevant information:

- YES → Proceed to answer with local context
- NO → Fall back to web search

### 4. Web Search (WEB_SEARCH Node)

If local documents aren't relevant:

- Searches the web using Tavily (primary) or DuckDuckGo (fallback)
- Extracts relevant snippets
- Uses web content as context

### 5. Answer Generation (ANSWER Node)

The LLM generates the final answer:

- Uses context from either local KB or web search
- Provides source attribution
- Maintains conversation quality

### 6. Reflection (REFLECT Node)

The agent evaluates its own answer:

- Is the answer relevant to the question?
- Is the answer complete?
- Confidence score (0-1)
- Strengths and weaknesses
- Missing information

## 🛠️ Troubleshooting

### Issue: Import errors or module not found

**Solution:** Make sure you're in the virtual environment and all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Issue: OpenAI API key error

**Solution:** Check your `.env` file has the correct API key:

```env
OPENAI_API_KEY=sk-...
```

### Issue: ChromaDB persistence error

**Solution:** Delete the `chroma_db/` directory and reinitialize:

```bash
rm -rf chroma_db  # Linux/Mac
rmdir /s chroma_db  # Windows
```

### Issue: Web search not working

**Solution:**

- For Tavily: Add `TAVILY_API_KEY` to `.env`
- Falls back to DuckDuckGo (no API key needed)
- Check network connection

### Issue: RAGAS evaluation failing

**Solution:** Install with datasets support:

```bash
pip install ragas datasets
```

## 📚 Documentation

### Project Documentation

- **[README.md](README.md)** - Main documentation (you are here)
- **[PROJECT_EXPERIENCE.md](PROJECT_EXPERIENCE.md)** - 🆕 Comprehensive AI/ML project experience
  - Detailed challenges and solutions
  - Technical architecture deep-dive
  - Performance metrics and outcomes
  - Key learnings and best practices
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** - Evaluation metrics and testing
- **Architecture Guides** (if available):
  - ARCHITECTURE.md - System architecture details
  - LANGSMITH_GUIDE.md - LangSmith tracing setup
  - PROVIDER_SETUP_GUIDE.md - Multi-provider configuration

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain** - RAG pipeline framework
- **LangGraph** - Agent orchestration
- **ChromaDB** - Vector database
- **OpenAI** - GPT models and embeddings
- **Groq** - Fast LLM inference
- **NVIDIA** - High-quality embeddings
- **Tavily** - Web search API
- **RAGAS** - Evaluation metrics

---

**Multi-Provider Edition** - Supporting OpenAI, Groq, and NVIDIA for maximum flexibility! 🚀

## Evaluation Results (Sample)

Final Results Analysis: EXCELLENT Performance! 🌟

Overall Grade: 8.5/10 (Very Good!)

✅ What's Working GREAT:

1. BERTScore (Semantic Similarity) - EXCELLENT! ⭐⭐⭐⭐⭐

- Average F1: 0.867 (86.7%) - Outstanding semantic understanding!
- Q1 (Renewable energy): 0.916 (91.6%) - Near perfect!
- Q4 (Wind energy): 0.884 (88.4%) - Excellent!
- Q2 (Types): 0.864 (86.4%) - Very good!
- Interpretation: The agent understands and conveys the correct meaning even if wording differs from ground truth.

2. Answer Quality - High

- All answers are detailed, accurate, and well-structured
- Average length: 819 characters (good balance - not too short, not verbose)
- 100% context usage - Agent properly uses retrieved documents
- Professional formatting with bullet points and headers

3. ROUGE Scores - Good

- Average ROUGE-L: 0.278 (27.8%) - Good word overlap
- Best: Q2 (Types) = 0.392 (39.2%)
- Q1 (Renewable) = 0.379 (37.9%)

4. Technical Accuracy

- All renewable energy facts are correct
- Wind energy explanation is technically sound
- Solar benefits are comprehensive and accurate

⚠️ What Needs Attention:

1. BLEU Scores - Low (But Expected)

- Average BLEU-4: 0.065 (6.5%) - Low exact phrase matching
- Q1: 0.159 (best)
- Q3: 0.005 (very low)
- Why this is OK: BLEU measures exact word matching, designed for translation. The agent paraphrases correctly (high BERTScore proves this). For RAG systems, semantic similarity matters more than exact wording.
