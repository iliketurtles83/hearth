"""Registers the music tool; the implementation lives in the `music` package."""
import tools as _registry
from music import tool as _tool

_registry.register("music", _tool)
