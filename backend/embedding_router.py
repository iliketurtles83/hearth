from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
import numpy as np

log = logging.getLogger("assistant.embedding_router")


TOOL_CLASSES = ("none", "weather", "music", "code", "vision")
DIALOGUE_CLASSES = ("local", "cloud", "memory-augmented")

DEFAULT_TOOL_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "none": (
        "Hello there",
        "How are you today",
        "Tell me a short joke",
        "Summarize this paragraph",
        "Can you explain this concept",
        "What does this sentence mean",
        "Help me brainstorm names",
        "Rewrite this in plain language",
        "What are some pros and cons",
        "Give me a concise answer",
        "Thanks that helped",
        "Can we continue this conversation",
        "What is a good next step",
        "Please clarify that response",
        "I have another question",
    ),
    "weather": (
        "What is the weather in Seattle",
        "Give me the forecast for tomorrow",
        "Is it going to rain today",
        "Current temperature in Berlin",
        "How humid is it outside",
        "Will it snow this weekend",
        "Weather report for New York",
        "Do I need an umbrella today",
        "How windy is it right now",
        "Forecast for San Francisco tonight",
        "What is the high and low today",
        "Tell me the weather near me",
        "Chance of rain this afternoon",
        "Is it sunny in Lisbon",
        "Weekend weather outlook",
    ),
    "music": (
        "Play music by Radiohead",
        "Queue Miles Davis",
        "Pause the music",
        "Resume playback",
        "Skip to the next track",
        "What is currently playing",
        "Show me the queue",
        "Stop the music",
        "Play some jazz",
        "Add this song to the queue",
        "Shuffle my playlist",
        "Put on Daft Punk",
        "Lower the volume to forty",
        "Play songs from the nineties",
        "Queue the album Kind of Blue",
    ),
    "code": (
        "Write a Python function to parse CSV",
        "Implement a binary search in JavaScript",
        "Fix this bug in my auth module",
        "Refactor this class for readability",
        "Add unit tests for this function",
        "Explain how this code works",
        "Review this pull request diff",
        "Create a FastAPI endpoint",
        "Debug this stack trace",
        "How do I implement rate limiting",
        "Generate SQL migration script",
        "Rename this variable across files",
        "Why does this algorithm fail",
        "Optimize this query",
        "Create test cases for edge conditions",
    ),
    "vision": (
        "Describe this image",
        "What is in this picture",
        "Read the text in this screenshot",
        "Analyze this diagram",
        "Identify objects in this photo",
        "What does this chart show",
        "Extract text from this image",
        "Explain this graph image",
        "What do you see in this screenshot",
        "Summarize this visual",
        "Read this receipt image",
        "Interpret this plot",
        "Describe the photo details",
        "Can you OCR this picture",
        "Tell me what is shown in this image",
    ),
}

DEFAULT_DIALOGUE_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "local": (
        "Hello",
        "Thanks",
        "Rewrite this paragraph",
        "Give me a short summary",
        "Translate this sentence",
        "Make this friendlier",
        "Explain this briefly",
        "What does this word mean",
        "Give me three bullet points",
        "Can you simplify this text",
        "Draft a quick response",
        "Help me phrase this",
        "Fix grammar in this sentence",
        "Brainstorm a title",
        "Give me a concise answer",
    ),
    "cloud": (
        "Compare microservices and monoliths deeply",
        "Design a distributed rate limiter",
        "Analyze architectural tradeoffs for event sourcing",
        "Create a comprehensive implementation plan",
        "Give an in-depth explanation of CAP theorem",
        "Break down this complex system design",
        "Evaluate multiple approaches for scaling",
        "Provide a thorough technical analysis",
        "Discuss pros and cons in detail",
        "Design a resilient message queue architecture",
        "Provide step by step reasoning",
        "Deep dive into performance bottlenecks",
        "Compare security models across architectures",
        "Reason through this ambiguous requirement",
        "Map out a complete migration strategy",
    ),
    "memory-augmented": (
        "What is my name",
        "What city did I mention before",
        "Do you remember my preference",
        "You said this earlier can you repeat it",
        "What did we decide last time",
        "Recall my favorite music genre",
        "Use the notes from our previous chat",
        "I mentioned this before",
        "Can you remember my setting",
        "What was the plan we discussed",
        "Please use my prior context",
        "Earlier you told me something",
        "What did I say about this previously",
        "Reference our past conversation",
        "Use my saved preferences",
    ),
}


@dataclass(frozen=True)
class ClassifierResult:
    label: str
    score: float
    second_label: str
    second_score: float
    gap: float
    ambiguous: bool


@dataclass(frozen=True)
class DualClassifierResult:
    tool: ClassifierResult
    dialogue: ClassifierResult
    should_escalate: bool


@dataclass(frozen=True)
class ExemplarIndex:
    classes: tuple[str, ...]
    row_labels: tuple[str, ...]
    row_texts: tuple[str, ...]
    matrix: np.ndarray

    @classmethod
    def from_embeddings(
        cls,
        exemplars: dict[str, tuple[str, ...]],
        embeddings: list[list[float]],
    ) -> "ExemplarIndex":
        row_labels: list[str] = []
        row_texts: list[str] = []
        ordered_classes = tuple(exemplars.keys())
        expected_rows = sum(len(v) for v in exemplars.values())
        if len(embeddings) != expected_rows:
            raise ValueError(
                f"Embedding row count mismatch: got {len(embeddings)}, expected {expected_rows}"
            )

        emb_i = 0
        for label, samples in exemplars.items():
            for sample in samples:
                row_labels.append(label)
                row_texts.append(sample)
                emb_i += 1

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
            raise ValueError("Embeddings must be a 2D matrix with one row per exemplar")
        matrix = _normalize_rows(matrix)

        return cls(
            classes=ordered_classes,
            row_labels=tuple(row_labels),
            row_texts=tuple(row_texts),
            matrix=matrix,
        )


@dataclass(frozen=True)
class EmbeddingRouterSnapshot:
    model: str
    dim: int
    created_at_unix: float
    tool_rows: int
    dialogue_rows: int


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return vector / norm


def _classify_index(
    query_embedding: np.ndarray,
    index: ExemplarIndex,
    *,
    ambiguity_gap: float,
    min_similarity: float,
) -> ClassifierResult:
    query = _normalize_vector(query_embedding.astype(np.float32, copy=False))

    # Single matrix multiply to score against every exemplar row.
    similarities = index.matrix @ query

    class_scores: list[tuple[str, float]] = []
    row_labels = np.asarray(index.row_labels, dtype=object)
    for label in index.classes:
        label_scores = similarities[row_labels == label]
        if label_scores.size == 0:
            continue
        class_scores.append((label, float(np.max(label_scores))))

    if not class_scores:
        raise ValueError("Classifier index has no scores")

    class_scores.sort(key=lambda item: item[1], reverse=True)
    top_label, top_score = class_scores[0]
    second_label, second_score = class_scores[1] if len(class_scores) > 1 else (top_label, -1.0)
    gap = top_score - second_score
    ambiguous = (gap < ambiguity_gap) or (top_score < min_similarity)

    return ClassifierResult(
        label=top_label,
        score=top_score,
        second_label=second_label,
        second_score=second_score,
        gap=gap,
        ambiguous=ambiguous,
    )


class EmbeddingIntentRouter:
    def __init__(
        self,
        *,
        tool_index: ExemplarIndex,
        dialogue_index: ExemplarIndex,
        tool_ambiguity_gap: float = 0.035,
        dialogue_ambiguity_gap: float = 0.03,
        tool_min_similarity: float = 0.15,
        dialogue_min_similarity: float = 0.15,
    ) -> None:
        self.tool_index = tool_index
        self.dialogue_index = dialogue_index
        self.tool_ambiguity_gap = tool_ambiguity_gap
        self.dialogue_ambiguity_gap = dialogue_ambiguity_gap
        self.tool_min_similarity = tool_min_similarity
        self.dialogue_min_similarity = dialogue_min_similarity

    def classify_embedding(self, query_embedding: np.ndarray) -> DualClassifierResult:
        tool = _classify_index(
            query_embedding,
            self.tool_index,
            ambiguity_gap=self.tool_ambiguity_gap,
            min_similarity=self.tool_min_similarity,
        )
        dialogue = _classify_index(
            query_embedding,
            self.dialogue_index,
            ambiguity_gap=self.dialogue_ambiguity_gap,
            min_similarity=self.dialogue_min_similarity,
        )
        return DualClassifierResult(
            tool=tool,
            dialogue=dialogue,
            should_escalate=(tool.ambiguous or dialogue.ambiguous),
        )

    def classify_text(self, text: str, embed_text: Callable[[str], np.ndarray]) -> DualClassifierResult:
        embedding = embed_text(text)
        if embedding.ndim != 1:
            raise ValueError("Text embedding must be a 1D vector")
        return self.classify_embedding(embedding)


async def ollama_embed_text(
    text: str,
    *,
    base_url: str,
    model: str,
    timeout_seconds: float = 8.0,
) -> np.ndarray:
    payload = {"model": model, "prompt": text}
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(f"{base_url.rstrip('/')}/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
    vector = np.asarray(data.get("embedding", []), dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Invalid embedding payload from Ollama")
    return vector


async def build_embedding_router(
    *,
    tool_exemplars: dict[str, tuple[str, ...]] | None = None,
    dialogue_exemplars: dict[str, tuple[str, ...]] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 8.0,
) -> tuple[EmbeddingIntentRouter, EmbeddingRouterSnapshot]:
    tool_bank = tool_exemplars or DEFAULT_TOOL_EXEMPLARS
    dialogue_bank = dialogue_exemplars or DEFAULT_DIALOGUE_EXEMPLARS

    ollama_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
    embed_model = model or os.getenv("ROUTER_EMBED_MODEL", "nomic-embed-text")

    all_texts: list[str] = []
    for samples in tool_bank.values():
        all_texts.extend(samples)
    for samples in dialogue_bank.values():
        all_texts.extend(samples)

    rows: list[list[float]] = []
    for text in all_texts:
        vec = await ollama_embed_text(
            text,
            base_url=ollama_url,
            model=embed_model,
            timeout_seconds=timeout_seconds,
        )
        rows.append(vec.tolist())

    tool_rows = sum(len(samples) for samples in tool_bank.values())
    tool_matrix_rows = rows[:tool_rows]
    dialogue_matrix_rows = rows[tool_rows:]

    tool_index = ExemplarIndex.from_embeddings(tool_bank, tool_matrix_rows)
    dialogue_index = ExemplarIndex.from_embeddings(dialogue_bank, dialogue_matrix_rows)
    router = EmbeddingIntentRouter(tool_index=tool_index, dialogue_index=dialogue_index)
    snapshot = EmbeddingRouterSnapshot(
        model=embed_model,
        dim=int(tool_index.matrix.shape[1]),
        created_at_unix=time.time(),
        tool_rows=int(tool_index.matrix.shape[0]),
        dialogue_rows=int(dialogue_index.matrix.shape[0]),
    )
    return router, snapshot


_router_cache: EmbeddingIntentRouter | None = None
_router_snapshot: EmbeddingRouterSnapshot | None = None
_router_error: str = ""
_router_lock = asyncio.Lock()


def get_embedding_router() -> EmbeddingIntentRouter | None:
    return _router_cache


def get_embedding_router_snapshot() -> EmbeddingRouterSnapshot | None:
    return _router_snapshot


def get_embedding_router_error() -> str:
    return _router_error


def embedding_router_ready() -> bool:
    return _router_cache is not None


async def warmup_embedding_router(
    *,
    force_refresh: bool = False,
    router_factory: Callable[..., Awaitable[tuple[EmbeddingIntentRouter, EmbeddingRouterSnapshot]]] = build_embedding_router,
) -> bool:
    global _router_cache, _router_snapshot, _router_error

    if _router_cache is not None and not force_refresh:
        return True

    async with _router_lock:
        if _router_cache is not None and not force_refresh:
            return True

        started = time.time()
        try:
            router, snapshot = await router_factory()
        except Exception as exc:
            _router_error = str(exc)
            if force_refresh:
                _router_cache = None
                _router_snapshot = None
            log.warning("embedding_router.warmup_failed | error=%s", exc)
            return False

        _router_cache = router
        _router_snapshot = snapshot
        _router_error = ""
        elapsed_ms = int((time.time() - started) * 1000)
        log.info(
            "embedding_router.warmup_ready | model=%s dim=%d tool_rows=%d dialogue_rows=%d elapsed_ms=%d",
            snapshot.model,
            snapshot.dim,
            snapshot.tool_rows,
            snapshot.dialogue_rows,
            elapsed_ms,
        )
        return True