"""Generated Cohere pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""

from typing import Literal

type CohereEmbedV4ModelId = Literal["cohere.embed-v4:0"]
"""Cataloged unprefixed Cohere Embed v4 identifiers for Amazon Bedrock."""

type CohereEmbedV3ModelId = Literal[
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
]
"""Cataloged unprefixed Cohere Embed v3 identifiers for Amazon Bedrock."""

type _CohereBedrockEmbeddingModelId = CohereEmbedV4ModelId | CohereEmbedV3ModelId

_COHERE_EMBED_V4_MODELS: frozenset[CohereEmbedV4ModelId] = frozenset({
    "cohere.embed-v4:0",
})
_COHERE_EMBED_V3_MODELS: frozenset[CohereEmbedV3ModelId] = frozenset({
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
})

COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS: dict[str, float] = {
    "cohere.embed-v4:0": 0.12,
    "cohere.embed-english-v3": 0.1,
    "cohere.embed-multilingual-v3": 0.1,
}
"""USD per million input tokens of each cataloged unprefixed Cohere Bedrock embedding model.

LiteLLM lists no other nonzero rate that a langchaint embedding request incurs.
"""

COHERE_BEDROCK_CROSS_REGION_MULTIPLIER: dict[str, float] = {
    "us": 1.0,
    "eu": float("nan"),
    "au": float("nan"),
    "jp": float("nan"),
    "apac": float("nan"),
    "global": 1.0,
    "us-gov": float("nan"),
}
"""Input-token rate multiplier of a Cohere model for each Bedrock cross-region prefix.

It is NaN for a prefix that LiteLLM prices for no Cohere model.
"""
