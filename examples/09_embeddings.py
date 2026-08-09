"""Create retrieval embeddings and reuse their account for generation."""

import numpy as np

from langchaint import Response
from langchaint.openai import OpenAIAccount


async def retrieve_and_summarize() -> Response[str]:
    """Embed documents, select one, and summarize it.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        Exception: Embedding or generation failed.
    """
    account = OpenAIAccount()
    embedding_model = account.embedding_model("text-embedding-3-small", dimension=256)
    summarizer = account.model("gpt-5.6-terra").bind(automatic_prompt_caching=False)

    documents = [
        "Whales are mammals that breathe air.",
        "Octopuses have three hearts.",
        "Sea turtles navigate across ocean basins.",
    ]
    document_vectors = await embedding_model.embed(documents, task="retrieval_document")
    query_vectors = await embedding_model.embed(
        ["Which animal has several hearts?"], task="retrieval_query"
    )

    print(f"documents shape: {document_vectors.shape}")
    print(f"query shape: {query_vectors.shape}")
    print(f"document norms: {np.linalg.vector_norm(document_vectors, axis=1)}")
    scores: np.ndarray[tuple[int], np.dtype[np.float32]] = document_vectors @ query_vectors[0]
    selected_document = documents[int(scores.argmax())]
    return await summarizer.generate_one(f"Summarize this fact: {selected_document}")
