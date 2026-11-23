import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag_assistant.generation.llm_client import OpenAICompatibleClient
from rag_assistant.evaluation.metrics import RAGEvaluator

def main():
    # SETUP GROK CLIENT
    # Replace with your actual Grok API info
    grok_client = OpenAICompatibleClient(
        api_key="YOUR_GROK_API_KEY",
        base_url="https://api.x.ai/v1", # Typical base URL for Grok
        model="grok-beta"
    )

    evaluator = RAGEvaluator(grok_client)

    # MOCK DATA FOR TESTING
    sample_query = "What is the remote work policy?"
    sample_context = "Employees can work remotely 3 days a week. All work must be coordinated with managers."
    sample_answer = "You can work from home 3 days a week, but you need to talk to your manager first."

    print("--- Running Evaluation ---")
    results = evaluator.evaluate_response(sample_query, sample_context, sample_answer)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import json
    main()
