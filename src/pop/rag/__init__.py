"""RAG stack: retrievers, prompt building, and batch generation.

Retrieval-augmented prompting of an instruction-tuned code LLM: retrieve
similar buggy->fixed exemplars, build a chat-formatted few-shot prompt, and
generate the fix. See ``pop.rag.prompt`` and ``pop.rag.retrievers``.
"""
