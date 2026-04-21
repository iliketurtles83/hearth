import os

LOCAL_MODEL = os.getenv("MODEL_LOCAL", "llama3.2")
CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")
CLOUD_THRESHOLD = int(os.getenv("CLOUD_THRESHOLD", "300"))

def should_use_cloud(prompt: str) -> bool:
    """Simple heuristic for now — grows into LangGraph in Phase 5."""
    if len(prompt) > CLOUD_THRESHOLD:
        return True
    cloud_keywords = ["explain in detail", "write a full", "architecture", "compare"]
    return any(kw in prompt.lower() for kw in cloud_keywords)