from typing import Dict
import structlog

logger = structlog.get_logger()

# Prices per 1M tokens (Approximate - update as needed)
MODEL_PRICES = {
    "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
    "grok-beta": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def log_request(self, model: str, input_tokens: int, output_tokens: int):
        prices = MODEL_PRICES.get(model, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        request_cost = input_cost + output_cost

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += request_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": request_cost
        }

    def get_summary(self):
        return {
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4)
        }
