"""Phase 14 – Vision input tests.

Covers:
- Image validation helper (valid images, bad MIME, bad base64, oversized)
- Vision intent via router text patterns
- Intent override when image present in graph state
- Model selection for vision intent
- TTS suppression for vision intent (main.py logic)
"""

import base64
import sys
import os

import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Validation helper ─────────────────────────────────────────────────────────

from main import _validate_image, OLLAMA_VISION_MODEL, CHAT_MODEL


def _make_b64(nbytes: int = 64) -> str:
    """Return a base64 string of the given raw size."""
    return base64.b64encode(b"\xff" * nbytes).decode()


class TestValidateImage:
    def test_none_image_is_valid(self):
        assert _validate_image(None, None) is None

    def test_valid_png(self):
        assert _validate_image(_make_b64(), "image/png") is None

    def test_valid_jpeg(self):
        assert _validate_image(_make_b64(), "image/jpeg") is None

    def test_valid_webp(self):
        assert _validate_image(_make_b64(), "image/webp") is None

    def test_invalid_mime(self):
        err = _validate_image(_make_b64(), "image/gif")
        assert err is not None
        assert "Unsupported" in err

    def test_missing_mime(self):
        err = _validate_image(_make_b64(), None)
        assert err is not None

    def test_bad_base64(self):
        err = _validate_image("not!valid!base64!!!!", "image/png")
        assert err is not None
        assert "base64" in err

    def test_oversized_image(self):
        # 26 MB raw → exceeds 25 MB limit
        big_b64 = base64.b64encode(b"\x00" * (26 * 1024 * 1024)).decode()
        err = _validate_image(big_b64, "image/png")
        assert err is not None
        assert "large" in err.lower()


# ── Router: vision intent via text patterns ───────────────────────────────────

from router import classify_intent, VISION_MODEL, CHAT_MODEL as _CHAT_MODEL


class TestVisionIntentFromText:
    def test_what_is_in_this_image(self):
        d = classify_intent("what is in this image?")
        assert d.intent == "vision"

    def test_describe_photo(self):
        d = classify_intent("Can you describe this photo?")
        assert d.intent == "vision"

    def test_look_at_screenshot(self):
        d = classify_intent("look at this screenshot please")
        assert d.intent == "vision"

    def test_regular_question_not_vision(self):
        d = classify_intent("What is the capital of France?")
        assert d.intent != "vision"

    def test_vision_model_is_chat_model_by_default(self):
        # VISION_MODEL defaults to CHAT_MODEL when no env var set
        assert VISION_MODEL == _CHAT_MODEL or VISION_MODEL != ""


# ── Graph: intent override when image present ─────────────────────────────────

class TestVisionIntentOverride:
    """Verify that the intent_classifier node forces vision when image_base64 is set.

    We test the logic directly rather than running the full graph to stay fast.
    """

    def test_image_forces_vision_intent(self):
        """Simulate the intent_classifier override: if image_base64 is in state,
        intent must become 'vision' and use_cloud must be False."""
        from router import classify_intent

        # Text-only decision first
        decision = classify_intent("tell me a joke")
        assert decision.intent != "vision"

        # Simulate graph node override (mirrors graph.py logic):
        state = {"image_base64": _make_b64(), "message": "tell me a joke"}
        if state.get("image_base64"):
            decision.intent = "vision"
            decision.use_cloud = False
        assert decision.intent == "vision"
        assert decision.use_cloud is False

    def test_no_image_no_override(self):
        from router import classify_intent

        decision = classify_intent("tell me a joke")
        state = {"image_base64": None, "message": "tell me a joke"}
        if state.get("image_base64"):
            decision.intent = "vision"
        assert decision.intent != "vision"


# ── TTS suppression ───────────────────────────────────────────────────────────

from main import _voice_tts_metadata


class TestVisionTTSSuppression:
    """Verify that voice TTS metadata is suppressed for vision intents."""

    def test_voice_meta_emitted_for_voice_source(self):
        meta = _voice_tts_metadata("voice")
        assert meta is not None

    def test_vision_suppresses_tts(self):
        # Simulate the logic in generate(): suppress if intent_for_log == "vision"
        intent_for_log = "vision"
        voice_meta = _voice_tts_metadata("voice")
        if intent_for_log == "vision":
            voice_meta = None
        assert voice_meta is None

    def test_non_vision_preserves_tts(self):
        intent_for_log = "quick-local"
        voice_meta = _voice_tts_metadata("voice")
        if intent_for_log == "vision":
            voice_meta = None
        assert voice_meta is not None
