import json
from typing import Dict, Any
from ..generation.llm_client import BaseLLMClient

EVAL_SYSTEM_PROMPT = """You are an expert AI Auditor. Your task is to evaluate RAG system responses.
You will be provided with a Question, the retrieved Context, and the generated Answer.
You must output your evaluation in valid JSON format with three scores (0.0 to 1.0) and a brief reasoning.

JSON SCHEMA:
{
    "faithfulness": 0.0,
    "relevance": 0.0,
    "context_precision": 0.0,
    "reasoning": "string"
}
"""

EVAL_USER_TEMPLATE = """QUESTION: {query}
CONTEXT: {context}
ANSWER: {answer}

Evaluate based on:
1. Faithfulness: Is the answer derived solely from the context?
2. Relevance: Does the answer directly address the question?
3. Context Precision: How much of the retrieved context was actually needed for the answer?
"""

class RAGEvaluator:
    def __init__(self, judge_client: BaseLLMClient):
        self.judge = judge_client

    def evaluate_response(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        prompt = EVAL_USER_TEMPLATE.format(
            query=query,
            context=context,
            answer=answer
        )

        raw_response = self.judge.generate(EVAL_SYSTEM_PROMPT, prompt)

        try:
            # Attempt to find JSON in the response if the model adds chatty text
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            json_str = raw_response[start:end]
            return json.loads(json_str)
        except Exception as e:
            return {
                "error": f"Failed to parse evaluation: {str(e)}",
                "raw_response": raw_response
            }
