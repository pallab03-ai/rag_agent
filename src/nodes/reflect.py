import logging
import json

from ..state import AgentState
from ..utils import (
    log_node_execution,
    format_error_message,
    get_env_variable,
    extract_confidence_score,
    parse_yes_no_response
)
from ..llm_factory import create_llm


logger = logging.getLogger("rag_agent")


def reflect_node(state: AgentState) -> AgentState:
    log_node_execution(logger, "REFLECT", state)
    
    try:
        question = state["question"]
        answer = state.get("answer", "")
        
        if not answer or not answer.strip():
            logger.warning("No answer to reflect on")
            state["reflection"] = {
                "is_relevant": False,
                "is_complete": False,
                "confidence": 0.0,
                "issues": ["No answer was generated"],
                "suggestions": ["Generate an answer first"]
            }
            return state
        
        logger.info("Reflecting on answer quality...")
        
        # Initialize LLM using factory
        llm = create_llm(temperature=0.0)  # Use low temperature for evaluation
        
        # Create reflection prompt
        reflection_prompt = f"""You are an answer quality evaluator. Analyze the provided answer to determine if it adequately addresses the user's question.

Question: {question}

Answer: {answer}

Provide your evaluation in the following JSON format:
{{
    "is_relevant": true/false,
    "is_complete": "yes/no/partial",
    "confidence": 0.85,
    "reasoning": "Brief explanation of your evaluation",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "missing_information": ["missing item 1", "missing item 2"],
    "overall_assessment": "Brief summary"
}}

Evaluation criteria:
- is_relevant: Does the answer address the question asked?
- is_complete: Is the answer comprehensive? (yes/no/partial)
- confidence: How confident are you in the answer quality? (0.0 to 1.0)
- strengths: What does the answer do well?
- weaknesses: What could be improved?
- missing_information: What important details are missing?

Respond ONLY with valid JSON, no additional text.
"""
        
        # Get LLM response
        logger.debug("Calling LLM for reflection...")
        response = llm.invoke(reflection_prompt)
        response_text = response.content.strip()
        
        logger.debug(f"Reflection response: {response_text[:200]}...")
        
        # Parse JSON response
        try:
            # Try to extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            reflection = json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reflection response as JSON: {e}")
            # Fallback: create basic reflection
            reflection = {
                "is_relevant": True,
                "is_complete": "partial",
                "confidence": 0.5,
                "reasoning": "Failed to parse reflection, defaulting to partial",
                "strengths": ["Answer was generated"],
                "weaknesses": ["Unable to evaluate properly"],
                "missing_information": [],
                "overall_assessment": "Evaluation incomplete"
            }
        
        # Extract and normalize reflection data
        is_relevant = reflection.get("is_relevant", True)
        is_complete = reflection.get("is_complete", "partial")
        confidence = reflection.get("confidence", 0.5)
        reasoning = reflection.get("reasoning", "No reasoning provided")
        strengths = reflection.get("strengths", [])
        weaknesses = reflection.get("weaknesses", [])
        missing_info = reflection.get("missing_information", [])
        assessment = reflection.get("overall_assessment", "")
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, float(confidence)))
        
        # Log reflection results
        logger.info(f"Reflection complete:")
        logger.info(f"  - Relevant: {is_relevant}")
        logger.info(f"  - Complete: {is_complete}")
        logger.info(f"  - Confidence: {confidence:.2f}")
        logger.info(f"  - Assessment: {assessment}")
        
        if weaknesses:
            logger.info(f"  - Weaknesses: {', '.join(weaknesses)}")
        
        if missing_info:
            logger.info(f"  - Missing info: {', '.join(missing_info)}")
        
        # Update state with reflection
        state["reflection"] = reflection
        
        logger.info("✓ Reflect node completed successfully")
        
        return state
        
    except Exception as e:
        error_msg = format_error_message(e, "Reflect node")
        logger.error(error_msg)
        state["error"] = error_msg
        # Provide default reflection on error
        state["reflection"] = {
            "is_relevant": True,
            "is_complete": "unknown",
            "confidence": 0.5,
            "reasoning": "Reflection failed due to error",
            "strengths": [],
            "weaknesses": ["Reflection process encountered an error"],
            "missing_information": [],
            "overall_assessment": f"Error during reflection: {str(e)}"
        }
        return state
