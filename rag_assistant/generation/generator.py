from typing import List, Dict, Any, Generator
from .prompts import RAG_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .llm_client import BaseLLMClient

class RAGGenerator:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        formatted = []
        # results are from ChromaDB.query()
        docs = search_results.get('documents', [[]])[0]
        metas = search_results.get('metadatas', [[]])[0]

        for doc, meta in zip(docs, metas):
            source = meta.get('file_name', 'Unknown')
            page = meta.get('page_count', 'N/A')
            formatted.append(f"SOURCE: {source} (Page: {page})\nCONTENT: {doc}\n")

        return "\n---\n".join(formatted)

    def answer(self, query: str, search_results: List[Dict[str, Any]], history: str = "") -> str:
        context = self._format_context(search_results)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            history=history,
            query=query
        )

        return self.llm_client.generate(RAG_SYSTEM_PROMPT, user_prompt)

    def answer_stream(self, query: str, search_results: List[Dict[str, Any]], history: str = "") -> Generator[str, None, None]:
        context = self._format_context(search_results)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            history=history,
            query=query
        )

        stream = self.llm_client.generate(RAG_SYSTEM_PROMPT, user_prompt, stream=True)

        for chunk in stream:
            # Handle Anthropic streaming
            if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                yield chunk.delta.text
            # Handle OpenAI/Grok streaming
            elif hasattr(chunk, 'choices') and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
