import logging
import json
from typing import Dict, Any

from ..state import AgentState
from ..utils import get_env_variable, log_node_execution, format_error_message
from ..llm_factory import create_llm


logger = logging.getLogger("rag_agent")


def plan_node(state: AgentState) -> AgentState:
    log_node_execution(logger, "PLAN", state)
    
    try:
        question = state["question"]
        
        if not question or not question.strip():
            logger.error("Empty question provided")
            state["error"] = "Empty question provided"
            state["needs_retrieval"] = False
            return state
        
        logger.info(f"Analyzing question: '{question[:100]}...'")
        
        # Initialize LLM using factory
        llm = create_llm(temperature=0.0)  # Use low temperature for planning
        
        # Create planning prompt
        planning_prompt = f"""Analyze the following question and determine if it requires retrieving information from a knowledge base about renewable energy, climate change, and sustainability.

Question: {question}

Provide your analysis in the following JSON format:
{{
    "needs_retrieval": true/false,
    "reasoning": "Brief explanation of why retrieval is or isn't needed",
    "search_terms": ["key", "terms", "to", "search"],
    "question_type": "factual/opinion/general_knowledge/greeting"
}}

Guidelines:
- Set needs_retrieval to TRUE if the question asks about:
  * Specific facts about renewable energy, climate change, or sustainability
  * Details that would be in technical documents
  * Information that requires domain knowledge
  * ANY factual question that could potentially be answered from documents
  
- Set needs_retrieval to FALSE ONLY if the question is:
  * A greeting or conversational query (hi, hello, how are you, thank you)
  * A request for capabilities (what can you do, who are you)
  
IMPORTANT: When in doubt, ALWAYS set needs_retrieval to TRUE. It's better to check the knowledge base even for general knowledge questions.

Respond ONLY with valid JSON, no additional text.
"""
        
        # Get LLM response
        logger.debug("Calling LLM for question analysis...")
        response = llm.invoke(planning_prompt)
        response_text = response.content.strip()
        
        logger.debug(f"LLM planning response: {response_text[:200]}...")
        
        # Parse JSON response
        try:
            # Try to extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            query_plan = json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: assume retrieval is needed for safety
            query_plan = {
                "needs_retrieval": True,
                "reasoning": "Failed to parse LLM response, defaulting to retrieval",
                "search_terms": [question],
                "question_type": "unknown"
            }
        
        # Extract planning results
        needs_retrieval = query_plan.get("needs_retrieval", True)
        reasoning = query_plan.get("reasoning", "No reasoning provided")
        search_terms = query_plan.get("search_terms", [question])
        question_type = query_plan.get("question_type", "unknown")
        
        logger.info(f"Planning complete:")
        logger.info(f"  - Question type: {question_type}")
        logger.info(f"  - Needs retrieval: {needs_retrieval}")
        logger.info(f"  - Reasoning: {reasoning}")
        logger.info(f"  - Search terms: {search_terms}")
        
        # Update state
        state["query_plan"] = query_plan
        state["needs_retrieval"] = needs_retrieval
        
        # Initialize other flags
        state["is_relevant"] = False
        state["needs_web_search"] = False
        
        logger.info("✓ Plan node completed successfully")
        
        return state
        
    except Exception as e:
        error_msg = format_error_message(e, "Plan node")
        logger.error(error_msg)
        state["error"] = error_msg
        state["needs_retrieval"] = False  # Fallback to no retrieval on error
        return state


def analyze_question_simple(question: str) -> bool:
    question_lower = question.lower().strip()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(question_lower.startswith(g) for g in greetings):
        return False
    
    if len(question_lower) < 5:
        return False
    
    domain_keywords = [
        "renewable", "energy", "solar", "wind", "hydro", "climate", "sustainability",
        "carbon", "emission", "green", "environment", "fossil", "battery", "power"
    ]
    if any(keyword in question_lower for keyword in domain_keywords):
        return True
    
    fact_indicators = ["what is", "how does", "why does", "explain", "describe", "tell me about"]
    if any(indicator in question_lower for indicator in fact_indicators):
        return True
    
    return True
