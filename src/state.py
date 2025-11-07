from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    question: str
    query_plan: Optional[Dict[str, Any]]
    needs_retrieval: bool
    retrieved_docs: Optional[List[str]]
    context: Optional[str]
    answer: Optional[str]
    reflection: Optional[Dict[str, Any]]
    is_relevant: bool
    needs_web_search: bool
    web_search_results: Optional[List[str]]
    source: Optional[str]
    error: Optional[str]


def create_initial_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        query_plan=None,
        needs_retrieval=False,
        retrieved_docs=None,
        context=None,
        answer=None,
        reflection=None,
        is_relevant=False,
        needs_web_search=False,
        web_search_results=None,
        source=None,
        error=None
    )
