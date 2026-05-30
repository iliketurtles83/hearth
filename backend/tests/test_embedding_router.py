import numpy as np
import pytest

import embedding_router as er


def _build_router(
    *,
    tool_gap: float = 0.05,
    dialogue_gap: float = 0.05,
    tool_floor: float = -1.0,
    dialogue_floor: float = -1.0,
) -> er.EmbeddingIntentRouter:
    tool_exemplars = {
        "none": ("hello",),
        "weather": ("forecast",),
        "music": ("play song",),
        "code": ("write function",),
        "vision": ("describe image",),
    }
    dialogue_exemplars = {
        "local": ("short response",),
        "cloud": ("deep architecture",),
        "memory-augmented": ("remember my preference",),
    }

    tool_embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ]
    dialogue_embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    tool_index = er.ExemplarIndex.from_embeddings(tool_exemplars, tool_embeddings)
    dialogue_index = er.ExemplarIndex.from_embeddings(dialogue_exemplars, dialogue_embeddings)
    return er.EmbeddingIntentRouter(
        tool_index=tool_index,
        dialogue_index=dialogue_index,
        tool_ambiguity_gap=tool_gap,
        dialogue_ambiguity_gap=dialogue_gap,
        tool_min_similarity=tool_floor,
        dialogue_min_similarity=dialogue_floor,
    )


def test_exemplar_index_row_count_validation() -> None:
    exemplars = {"none": ("a", "b")}
    with pytest.raises(ValueError, match="Embedding row count mismatch"):
        er.ExemplarIndex.from_embeddings(exemplars, [[1.0, 0.0]])


def test_classify_embedding_returns_top_labels() -> None:
    router = _build_router(tool_gap=0.005, dialogue_gap=0.02)
    result = router.classify_embedding(np.asarray([1.0, 0.0], dtype=np.float32))

    assert result.tool.label == "none"
    assert result.tool.second_label == "weather"
    assert result.tool.score > result.tool.second_score
    assert result.dialogue.label == "local"
    assert result.should_escalate is False


def test_tool_gap_ambiguity_triggers_escalation() -> None:
    router = _build_router(tool_gap=0.2, dialogue_gap=0.02)
    result = router.classify_embedding(np.asarray([1.0, 0.0], dtype=np.float32))

    assert result.tool.label == "none"
    assert result.tool.ambiguous is True
    assert result.should_escalate is True


def test_similarity_floor_marks_ambiguous() -> None:
    router = _build_router(tool_gap=0.02, dialogue_gap=0.02, tool_floor=0.995, dialogue_floor=-1.0)
    result = router.classify_embedding(np.asarray([0.8, 0.2], dtype=np.float32))

    assert result.tool.score < 0.995
    assert result.tool.ambiguous is True
    assert result.should_escalate is True


def test_router_uses_module_threshold_defaults() -> None:
    tool_exemplars = {"none": ("hello",)}
    dialogue_exemplars = {"local": ("hi",)}
    tool_index = er.ExemplarIndex.from_embeddings(tool_exemplars, [[1.0, 0.0]])
    dialogue_index = er.ExemplarIndex.from_embeddings(dialogue_exemplars, [[1.0, 0.0]])
    router = er.EmbeddingIntentRouter(tool_index=tool_index, dialogue_index=dialogue_index)

    assert router.tool_ambiguity_gap == er.TOOL_AMBIGUITY_GAP_DEFAULT
    assert router.dialogue_ambiguity_gap == er.DIALOGUE_AMBIGUITY_GAP_DEFAULT
    assert router.tool_min_similarity == er.TOOL_MIN_SIMILARITY_DEFAULT
    assert router.dialogue_min_similarity == er.DIALOGUE_MIN_SIMILARITY_DEFAULT


def test_snapshot_model_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _build_router(tool_gap=0.02, dialogue_gap=0.02)
    router.snapshot_model = "nomic-embed-text"
    monkeypatch.setenv("ROUTER_EMBED_MODEL", "mxbai-embed-large")

    with pytest.raises(er.EmbeddingRouterSnapshotMismatchError, match="snapshot model mismatch"):
        router.classify_embedding(np.asarray([1.0, 0.0], dtype=np.float32))


def test_classify_text_uses_embed_callback() -> None:
    router = _build_router(tool_gap=0.02, dialogue_gap=0.02)

    def _embed(_text: str) -> np.ndarray:
        return np.asarray([0.0, 1.0], dtype=np.float32)

    result = router.classify_text("play something", _embed)
    assert result.tool.label == "music"
    assert result.dialogue.label == "cloud"


@pytest.mark.asyncio
async def test_warmup_embedding_router_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_exemplars = {"none": ("hello",)}
    dialogue_exemplars = {"local": ("hi",)}
    tool_index = er.ExemplarIndex.from_embeddings(tool_exemplars, [[1.0, 0.0]])
    dialogue_index = er.ExemplarIndex.from_embeddings(dialogue_exemplars, [[1.0, 0.0]])
    router = er.EmbeddingIntentRouter(tool_index=tool_index, dialogue_index=dialogue_index)
    snapshot = er.EmbeddingRouterSnapshot(
        model="nomic-embed-text",
        dim=2,
        created_at_unix=0.0,
        tool_rows=1,
        dialogue_rows=1,
    )

    call_count = {"n": 0}

    async def _factory():
        call_count["n"] += 1
        return router, snapshot

    monkeypatch.setattr(er, "_router_cache", None)
    monkeypatch.setattr(er, "_router_snapshot", None)
    monkeypatch.setattr(er, "_router_error", "")

    ok_first = await er.warmup_embedding_router(router_factory=_factory)
    ok_second = await er.warmup_embedding_router(router_factory=_factory)

    assert ok_first is True
    assert ok_second is True
    assert call_count["n"] == 1
    assert er.embedding_router_ready() is True
    assert er.get_embedding_router_snapshot() == snapshot
    assert er.get_embedding_router_error() == ""


@pytest.mark.asyncio
async def test_warmup_embedding_router_failure_sets_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _failing_factory():
        raise RuntimeError("embedding init failed")

    monkeypatch.setattr(er, "_router_cache", None)
    monkeypatch.setattr(er, "_router_snapshot", None)
    monkeypatch.setattr(er, "_router_error", "")

    ok = await er.warmup_embedding_router(router_factory=_failing_factory, force_refresh=True)

    assert ok is False
    assert er.embedding_router_ready() is False
    assert "embedding init failed" in er.get_embedding_router_error()
    assert er.get_embedding_router_snapshot() is None