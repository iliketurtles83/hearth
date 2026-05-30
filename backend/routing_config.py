from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingConfig:
    """Runtime routing knobs loaded from environment.

    Fields:
    - chat_token_budget: token budget used for history selection.
    - chat_max_turns: max conversational turns retained for prompt building.
    - ollama_url: base URL for local model API calls.
    - route_confidence_threshold: minimum confidence for cloud reasoning routes.
    - router_embedding_enabled: enables embedding-based intent routing.
    - router_embed_model: embedding model name for query/exemplar embeddings.
    - router_embed_timeout_ms: timeout for embedding API calls.
    - router_embedding_warmup: warm embedding router snapshot on startup.
    """

    chat_token_budget: int
    chat_max_turns: int
    ollama_url: str
    route_confidence_threshold: float
    router_embedding_enabled: bool
    router_embed_model: str
    router_embed_timeout_ms: int
    router_embedding_warmup: bool


def load_routing_config() -> RoutingConfig:
    return RoutingConfig(
        chat_token_budget=int(os.getenv("CHAT_TOKEN_BUDGET", "1500")),
        chat_max_turns=int(os.getenv("CHAT_MAX_TURNS", "24")),
        ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/"),
        route_confidence_threshold=float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.55")),
        router_embedding_enabled=os.getenv("ROUTER_EMBEDDING_ENABLED", "true").lower() == "true",
        router_embed_model=os.getenv("ROUTER_EMBED_MODEL", "nomic-embed-text"),
        router_embed_timeout_ms=int(os.getenv("ROUTER_EMBED_TIMEOUT_MS", "1500")),
        router_embedding_warmup=os.getenv("ROUTER_EMBEDDING_WARMUP", "true").strip().lower() == "true",
    )


ROUTING_CONFIG = load_routing_config()
