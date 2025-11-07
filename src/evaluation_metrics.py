"""
Comprehensive RAG Evaluation with Multiple Metrics

This module provides evaluation using:
1. RAGAs metrics (Context Relevance, Faithfulness, Answer Relevance)
2. Traditional NLP metrics (BLEU, ROUGE, BERTScore)
3. LLM-as-Judge evaluation
4. Custom RAG-specific metrics
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = logging.getLogger("rag_evaluation")


class RAGEvaluator:
    """Comprehensive RAG evaluation with multiple metrics"""
    
    def __init__(self, use_ragas: bool = True, use_nlp_metrics: bool = True, llm=None, embeddings=None):
        """
        Initialize evaluator
        
        Args:
            use_ragas: Use RAGAs metrics (requires ragas package)
            use_nlp_metrics: Use NLP metrics (BLEU, ROUGE, BERTScore)
            llm: LangChain LLM to use for RAGAs (optional, will create one if not provided)
            embeddings: LangChain embeddings to use for RAGAs (optional)
        """
        self.use_ragas = use_ragas
        self.use_nlp_metrics = use_nlp_metrics
        
        # Import optional dependencies
        if use_ragas:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision
                )
                from datasets import Dataset
                
                self.ragas_evaluate = evaluate
                self.ragas_metrics = {
                    'faithfulness': faithfulness,
                    'answer_relevancy': answer_relevancy,
                    'context_recall': context_recall,
                    'context_precision': context_precision
                }
                self.Dataset = Dataset
                
                # Set up LLM for RAGAs
                if llm is None:
                    # Try to create LLM from environment
                    llm = self._create_llm_for_ragas()
                
                if llm:
                    # Configure RAGAs to use the provided LLM
                    for metric in self.ragas_metrics.values():
                        if hasattr(metric, 'llm'):
                            metric.llm = llm
                        if hasattr(metric, 'embeddings') and embeddings:
                            metric.embeddings = embeddings
                    logger.info("RAGAs metrics loaded and configured with LLM")
                else:
                    logger.warning("No LLM configured for RAGAs, metrics may not work properly")
                    
            except ImportError as e:
                logger.warning(f"RAGAs not installed. Install with: pip install ragas. Error: {e}")
                self.use_ragas = False
            except Exception as e:
                logger.warning(f"RAGAs loading failed: {e}")
                self.use_ragas = False
        
        if use_nlp_metrics:
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                from rouge_score import rouge_scorer
                from bert_score import score as bert_score
                
                self.sentence_bleu = sentence_bleu
                self.SmoothingFunction = SmoothingFunction
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                self.bert_score = bert_score
                logger.info("✓ NLP metrics loaded")
            except ImportError:
                logger.warning("NLP metrics not fully available. Install with: pip install nltk rouge-score bert-score")
                self.use_nlp_metrics = False
    
    def _create_llm_for_ragas(self):
        """Create an LLM for RAGAs evaluation from environment variables"""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check if OpenAI key exists for RAGAs (required for embeddings)
            has_openai = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
            
            if not has_openai:
                logger.warning("RAGAs requires OPENAI_API_KEY for embeddings. Set it in .env to enable RAGAs metrics.")
                logger.info("You can still use GROQ_API_KEY for the agent, but RAGAs needs OpenAI.")
                return None
            
            # Use OpenAI for both LLM and embeddings (most compatible with RAGAs)
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("Created OpenAI LLM for RAGAs")
            return llm
                
        except Exception as e:
            logger.warning(f"Failed to create LLM for RAGAs: {e}")
            return None
    
    def evaluate_with_ragas(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate using RAGAs metrics
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of documents for each question)
            ground_truths: Optional ground truth answers
        
        Returns:
            Dictionary of metric scores
        """
        if not self.use_ragas:
            logger.warning("RAGAs not available")
            return {}
        
        try:
            # Prepare dataset
            data = {
                'question': questions,
                'answer': answers,
                'contexts': contexts,
            }
            
            if ground_truths:
                data['ground_truth'] = ground_truths
            
            dataset = self.Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics_to_use = [
                self.ragas_metrics['faithfulness'],
                self.ragas_metrics['answer_relevancy']
            ]
            
            if ground_truths:
                metrics_to_use.extend([
                    self.ragas_metrics['context_recall'],
                    self.ragas_metrics['context_precision']
                ])
            
            # Evaluate
            logger.info("Running RAGAs evaluation...")
            
            # RAGAs needs llm and embeddings configured globally
            import os
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("RAGAs requires OPENAI_API_KEY to be set for embeddings")
                logger.warning("Skipping RAGAs evaluation. Set OPENAI_API_KEY in .env to enable.")
                return {}
            
            result = self.ragas_evaluate(dataset, metrics=metrics_to_use)
            
            # Extract scores - handle both dict and EvaluationResult object
            if hasattr(result, 'to_pandas'):
                # New RAGAs version returns EvaluationResult
                df = result.to_pandas()
                # Get only numeric columns (the metric scores)
                numeric_cols = df.select_dtypes(include=['number']).columns
                scores = df[numeric_cols].mean().to_dict()
            elif isinstance(result, dict):
                # Old version returns dict
                scores = {k: v for k, v in result.items() if isinstance(v, (int, float))}
            else:
                # Try converting to dict
                scores = dict(result)
            
            logger.info(f"RAGAs evaluation complete: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def evaluate_with_nlp_metrics(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Evaluate using traditional NLP metrics
        
        Args:
            generated: Generated answer
            reference: Reference/ground truth answer
        
        Returns:
            Dictionary of metric scores
        """
        if not self.use_nlp_metrics:
            logger.warning("NLP metrics not available")
            return {}
        
        scores = {}
        
        try:
            # BLEU Score
            reference_tokens = reference.split()
            generated_tokens = generated.split()
            smoothing = self.SmoothingFunction().method1
            
            bleu1 = self.sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = self.sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu4 = self.sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            scores['bleu1'] = bleu1
            scores['bleu2'] = bleu2
            scores['bleu4'] = bleu4
            
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
        
        try:
            # ROUGE Scores
            rouge_scores = self.rouge_scorer.score(reference, generated)
            scores['rouge1_f'] = rouge_scores['rouge1'].fmeasure
            scores['rouge2_f'] = rouge_scores['rouge2'].fmeasure
            scores['rougeL_f'] = rouge_scores['rougeL'].fmeasure
            
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
        
        try:
            # BERTScore
            P, R, F1 = self.bert_score([generated], [reference], lang='en', verbose=False)
            scores['bertscore_precision'] = P.mean().item()
            scores['bertscore_recall'] = R.mean().item()
            scores['bertscore_f1'] = F1.mean().item()
            
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
        
        return scores
    
    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single answer
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional reference answer
        
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {
            'question': question,
            'answer': answer,
            'has_ground_truth': ground_truth is not None
        }
        
        # Custom metrics
        results['answer_length'] = len(answer)
        results['context_length'] = len(context)
        results['answer_has_context'] = context.lower() in answer.lower() or any(
            word in answer.lower() for word in context.lower().split()[:20]
        )
        
        # NLP metrics (if ground truth available)
        if ground_truth and self.use_nlp_metrics:
            nlp_scores = self.evaluate_with_nlp_metrics(answer, ground_truth)
            results['nlp_metrics'] = nlp_scores
        
        return results
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases
        
        Args:
            test_cases: List of test cases with format:
                {
                    'question': str,
                    'answer': str,
                    'contexts': List[str],
                    'ground_truth': str (optional)
                }
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        # Prepare data for RAGAs
        questions = [tc['question'] for tc in test_cases]
        answers = [tc['answer'] for tc in test_cases]
        contexts = [tc['contexts'] for tc in test_cases]
        ground_truths = [tc.get('ground_truth') for tc in test_cases]
        
        # Check if all test cases have ground truth
        has_ground_truth = all(gt is not None for gt in ground_truths)
        
        results = {
            'total_cases': len(test_cases),
            'individual_results': [],
            'aggregate_metrics': {}
        }
        
        # RAGAs evaluation
        if self.use_ragas and len(questions) > 0:
            ragas_scores = self.evaluate_with_ragas(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths if has_ground_truth else None
            )
            results['ragas_metrics'] = ragas_scores
        
        # Individual evaluations
        for tc in test_cases:
            eval_result = self.evaluate_answer_quality(
                question=tc['question'],
                answer=tc['answer'],
                context='\n'.join(tc['contexts']),
                ground_truth=tc.get('ground_truth')
            )
            results['individual_results'].append(eval_result)
        
        # Aggregate custom metrics
        if results['individual_results']:
            avg_answer_length = sum(r['answer_length'] for r in results['individual_results']) / len(results['individual_results'])
            context_usage_rate = sum(1 for r in results['individual_results'] if r['answer_has_context']) / len(results['individual_results'])
            
            results['aggregate_metrics'] = {
                'avg_answer_length': avg_answer_length,
                'context_usage_rate': context_usage_rate
            }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nTotal Test Cases: {results['total_cases']}")
        
        # RAGAs metrics
        if 'ragas_metrics' in results:
            print("\n" + "-"*80)
            print("RAGAs METRICS")
            print("-"*80)
            for metric, score in results['ragas_metrics'].items():
                print(f"  {metric}: {score:.4f}")
        
        # Aggregate metrics
        if 'aggregate_metrics' in results:
            print("\n" + "-"*80)
            print("AGGREGATE METRICS")
            print("-"*80)
            for metric, value in results['aggregate_metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save results to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")


def install_dependencies():
    """Install required packages for evaluation"""
    packages = [
        "ragas",
        "nltk",
        "rouge-score",
        "bert-score",
        "datasets"
    ]
    
    print("Installing evaluation packages...")
    print("Run this command:")
    print(f"pip install {' '.join(packages)}")


if __name__ == "__main__":
    print("RAG Evaluation Module")
    print("="*80)
    print("\nTo use this module, first install dependencies:")
    install_dependencies()
    print("\nThen import and use RAGEvaluator class in your code.")
