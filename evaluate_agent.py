"""
RAG Agent Evaluation Script

This script evaluates the RAG agent's performance on various metrics:
- RAGAs metrics (Faithfulness, Answer Relevancy, Context Relevancy, etc.)
- Traditional NLP metrics (BLEU, ROUGE, BERTScore)
- Retrieval accuracy
- Answer quality
- Response time
- Source attribution
- Relevance checking
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

from src.agent import RAGAgent
from src.rag_pipeline import RAGPipeline
from src.llm_factory import create_llm
from src.evaluation_metrics import RAGEvaluator

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluation")


class RAGEvaluator:
    """Evaluator for RAG Agent performance"""
    
    def __init__(self, agent: RAGAgent, llm_provider: str = "groq", llm_model: str = "openai/gpt-oss-120b"):
        """
        Initialize evaluator
        
        Args:
            agent: RAGAgent instance to evaluate
            llm_provider: LLM provider for evaluation
            llm_model: LLM model for evaluation
        """
        self.agent = agent
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.results = []
        
    def evaluate_test_set(self, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate agent on a test set of questions
        
        Args:
            test_questions: List of test cases with format:
                {
                    "question": "What is renewable energy?",
                    "expected_source": "local",  # local, web, or none
                    "category": "factual",  # factual, definition, comparison, etc.
                }
        
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Starting evaluation with {len(test_questions)} test cases...")
        
        total_tests = len(test_questions)
        correct_source = 0
        total_time = 0
        results_by_category = {}
        
        for i, test in enumerate(test_questions, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Test {i}/{total_tests}: {test['question']}")
            logger.info(f"{'='*80}")
            
            try:
                # Run agent
                start_time = time.time()
                result = self.agent.run(test['question'])
                elapsed_time = time.time() - start_time
                
                # Extract results
                answer = result.get('answer', '')
                source = result.get('source', 'none')
                retrieved_docs = result.get('retrieved_docs', [])
                web_results = result.get('web_search_results', [])
                
                # Check if source matches expected
                expected_source = test.get('expected_source', 'any')
                source_correct = (expected_source == 'any' or source == expected_source)
                
                if source_correct:
                    correct_source += 1
                
                # Evaluate answer quality
                answer_quality = self._evaluate_answer_quality(
                    question=test['question'],
                    answer=answer,
                    expected_answer=test.get('expected_answer', None)
                )
                
                # Store result
                test_result = {
                    'question': test['question'],
                    'category': test.get('category', 'unknown'),
                    'expected_source': expected_source,
                    'actual_source': source,
                    'source_correct': source_correct,
                    'answer': answer,
                    'answer_length': len(answer),
                    'num_retrieved_docs': len(retrieved_docs),
                    'num_web_results': len(web_results),
                    'response_time': elapsed_time,
                    'answer_quality': answer_quality,
                    'error': None
                }
                
                # Track by category
                category = test.get('category', 'unknown')
                if category not in results_by_category:
                    results_by_category[category] = []
                results_by_category[category].append(test_result)
                
                self.results.append(test_result)
                total_time += elapsed_time
                
                # Log result
                logger.info(f"✓ Source: {source} (Expected: {expected_source}) - {'CORRECT' if source_correct else 'WRONG'}")
                logger.info(f"✓ Response time: {elapsed_time:.2f}s")
                logger.info(f"✓ Answer quality: {answer_quality}")
                logger.info(f"✓ Answer length: {len(answer)} chars")
                
            except Exception as e:
                logger.error(f"✗ Error in test case: {e}")
                test_result = {
                    'question': test['question'],
                    'category': test.get('category', 'unknown'),
                    'error': str(e)
                }
                self.results.append(test_result)
        
        # Calculate metrics
        avg_time = total_time / total_tests if total_tests > 0 else 0
        source_accuracy = correct_source / total_tests if total_tests > 0 else 0
        
        # Calculate average answer quality
        quality_scores = [r.get('answer_quality', {}).get('relevance_score', 0) 
                         for r in self.results if 'answer_quality' in r]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        summary = {
            'total_tests': total_tests,
            'source_accuracy': source_accuracy,
            'correct_sources': correct_source,
            'average_response_time': avg_time,
            'average_answer_quality': avg_quality,
            'results_by_category': {
                cat: {
                    'count': len(results),
                    'avg_response_time': sum(r['response_time'] for r in results if 'response_time' in r) / len(results)
                }
                for cat, results in results_by_category.items()
            },
            'detailed_results': self.results
        }
        
        return summary
    
    def _evaluate_answer_quality(self, question: str, answer: str, expected_answer: str = None) -> Dict[str, Any]:
        """
        Evaluate answer quality using LLM
        
        Args:
            question: Original question
            answer: Agent's answer
            expected_answer: Optional expected answer for comparison
            
        Returns:
            Quality metrics dictionary
        """
        try:
            llm = create_llm(
                provider=self.llm_provider,
                model=self.llm_model,
                temperature=0.0
            )
            
            eval_prompt = f"""Evaluate the quality of this answer to the given question.

Question: {question}

Answer: {answer}

Evaluate on these criteria (score each 0-10):
1. Relevance: Does the answer address the question?
2. Completeness: Is the answer comprehensive?
3. Accuracy: Is the information correct? (assume yes if no contradictions)
4. Clarity: Is the answer clear and well-structured?

Respond in JSON format:
{{
    "relevance_score": 0-10,
    "completeness_score": 0-10,
    "accuracy_score": 0-10,
    "clarity_score": 0-10,
    "overall_score": 0-10,
    "feedback": "Brief evaluation summary"
}}
"""
            
            response = llm.invoke(eval_prompt)
            response_text = response.content.strip()
            
            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            quality_metrics = json.loads(response_text)
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Failed to evaluate answer quality: {e}")
            return {
                "relevance_score": 0,
                "completeness_score": 0,
                "accuracy_score": 0,
                "clarity_score": 0,
                "overall_score": 0,
                "feedback": f"Evaluation failed: {e}"
            }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Source Accuracy: {summary['source_accuracy']:.2%} ({summary['correct_sources']}/{summary['total_tests']})")
        print(f"Average Response Time: {summary['average_response_time']:.2f}s")
        print(f"Average Answer Quality: {summary['average_answer_quality']:.2f}/10")
        
        print("\n" + "-"*80)
        print("RESULTS BY CATEGORY")
        print("-"*80)
        for category, stats in summary['results_by_category'].items():
            print(f"\n{category.upper()}:")
            print(f"  Tests: {stats['count']}")
            print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
        
        print("\n" + "="*80)
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")


def create_test_set() -> List[Dict[str, Any]]:
    """
    Create a test set of questions for evaluation
    
    Returns:
        List of test questions
    """
    test_questions = [
        # Questions about documents (should use local KB)
        {
            "question": "What is renewable energy?",
            "expected_source": "local",
            "category": "definition",
            "expected_answer": "Renewable energy is energy derived from natural sources that are replenished at a higher rate than they are consumed. Examples include solar, wind, hydroelectric, geothermal, and biomass energy. These sources are sustainable and produce little to no greenhouse gas emissions during operation."
        },
        {
            "question": "What are the types of renewable energy?",
            "expected_source": "local",
            "category": "factual",
            "expected_answer": "The main types of renewable energy are: solar energy (from sunlight), wind energy (from air movement), hydroelectric energy (from flowing water), geothermal energy (from Earth's internal heat), and biomass energy (from organic materials like plants and waste)."
        },
        {
            "question": "What are the benefits of solar energy?",
            "expected_source": "local",
            "category": "factual",
            "expected_answer": "Solar energy benefits include: it's a clean renewable resource with no emissions during operation, it's abundant and free, it reduces electricity bills, requires minimal maintenance, costs have decreased significantly, it's scalable from small to large installations, creates jobs, and reduces dependence on fossil fuels."
        },
        {
            "question": "How does wind energy work?",
            "expected_source": "local",
            "category": "explanation",
            "expected_answer": "Wind energy works by using wind turbines to convert the kinetic energy of moving air into electricity. When wind blows, it turns the turbine blades, which spin a rotor connected to a generator. The generator then converts this mechanical energy into electrical energy that can be transmitted to the power grid."
        },
        
        # Questions NOT in documents (should use web or general knowledge)
        {
            "question": "Who is the current president of the United States?",
            "expected_source": "any",  # Could be web or none
            "category": "current_events",
            "expected_answer": "As of November 2025, Donald Trump is the President of the United States, having been inaugurated in January 2025 after winning the 2024 presidential election."
        },
        {
            "question": "What is the capital of France?",
            "expected_source": "any",
            "category": "general_knowledge",
            "expected_answer": "The capital of France is Paris."
        },
        
        # Greetings (should use none)
        {
            "question": "Hello, how are you?",
            "expected_source": "none",
            "category": "greeting",
            "expected_answer": "Hello! I'm doing well, thank you for asking. How can I help you today?"
        },
        {
            "question": "What can you help me with?",
            "expected_source": "none",
            "category": "meta",
            "expected_answer": "I can help you with questions about renewable energy and sustainability from my knowledge base, or search the web for current information. I can answer questions, provide explanations, and help you understand various topics."
        },
    ]
    
    return test_questions


def main():
    """Main evaluation function"""
    print("\n" + "="*80)
    print("RAG AGENT EVALUATION")
    print("="*80)
    
    # Initialize RAG Pipeline and Agent
    print("\n1. Initializing RAG system...")
    rag_pipeline = RAGPipeline()
    
    # Setup knowledge base
    print("\n2. Setting up knowledge base...")
    vector_store_manager = rag_pipeline.setup_knowledge_base(
        data_directory="data",
        reset=False,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "nvidia"),
        embedding_model=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/llama-3.2-nemoretriever-300m-embed-v1"),
        embedding_api_key=os.getenv("NVIDIA_API_KEY"),
        embedding_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    
    # Initialize agent
    print("\n3. Initializing agent...")
    agent = RAGAgent(
        vector_store_manager=vector_store_manager,
        enable_reflection=True
    )
    
    # Create evaluator
    print("\n4. Creating evaluator...")
    evaluator = RAGEvaluator(
        agent=agent,
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    
    # Create test set
    print("\n5. Loading test questions...")
    test_questions = create_test_set()
    print(f"   Loaded {len(test_questions)} test questions")
    
    # Run evaluation
    print("\n6. Running evaluation...")
    summary = evaluator.evaluate_test_set(test_questions)
    
    # Print summary
    evaluator.print_summary(summary)
    
    # Run comprehensive RAG evaluation with RAGAs
    print("\n7. Running comprehensive RAG evaluation (RAGAs, BLEU, ROUGE, BERTScore)...")
    run_comprehensive_evaluation(agent, test_questions)
    
    # Save results
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_{timestamp}.json"
    
    evaluator.save_results(str(results_file))
    
    # Save summary
    summary_file = results_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n+ Full results saved to: {results_file}")
    print(f"+ Summary saved to: {summary_file}")
    print("\nEvaluation complete!")


def run_comprehensive_evaluation(agent: RAGAgent, test_questions: List[Dict]):
    """
    Run comprehensive evaluation with RAGAs and NLP metrics
    
    Args:
        agent: RAGAgent instance
        test_questions: List of test questions
    """
    try:
        from src.evaluation_metrics import RAGEvaluator as ComprehensiveEvaluator
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION (RAGAs + NLP Metrics)")
        print("="*80)
        
        # Initialize comprehensive evaluator
        comp_evaluator = ComprehensiveEvaluator(use_ragas=True, use_nlp_metrics=True)
        
        # Run agent on test cases and collect results
        test_cases = []
        for test_q in test_questions[:5]:  # Evaluate first 5 for speed
            result = agent.run(test_q['question'])
            
            test_case = {
                'question': test_q['question'],
                'answer': result.get('answer', ''),
                'contexts': result.get('retrieved_docs', []),
                'ground_truth': test_q.get('expected_answer', None)
            }
            test_cases.append(test_case)
        
        # Run evaluation
        results = comp_evaluator.evaluate_batch(test_cases)
        
        # Print results
        comp_evaluator.print_results(results)
        
        # Save comprehensive results
        results_dir = Path("evaluation_results")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comprehensive_file = results_dir / f"comprehensive_eval_{timestamp}.json"
        comp_evaluator.save_results(results, str(comprehensive_file))
        
        print(f"\nComprehensive results saved to: {comprehensive_file}")
        
    except ImportError as e:
        print(f"\nWARNING: Comprehensive evaluation requires additional packages.")
        print(f"   Install with: pip install ragas nltk rouge-score bert-score datasets")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"\nWARNING: Comprehensive evaluation failed: {e}")


if __name__ == "__main__":
    main()
