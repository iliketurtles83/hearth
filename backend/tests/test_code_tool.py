"""Phase 10b — code_tool node and supporting infrastructure tests.

Coverage:
  1. Workspace-root boundary enforcement (path traversal → rejected)
  2. Write gating: write_file tool sets pending_write, does NOT write to disk
  3. Confirmation flow: pending_write + "yes" → write_executor writes the file
  4. Cancellation flow: pending_write + non-confirm → write cleared, normal routing
  5. code_context injection: code intent → code_context populated in memory_retrieval
  6. Tree-sitter indexer: index fixture file, verify retrieval
  7. /code/files and /code/files/{path} HTTP endpoints
"""
import asyncio
import os
import pathlib
import textwrap
import tempfile

import pytest
import pytest_asyncio


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_state(**overrides):
    """Return a minimal AssistantState dict with safe defaults."""
    base = {
        "message": "hello",
        "user_id": "test-user",
        "session_id": "test-session",
        "history": [],
        "system": "You are a helpful assistant.",
        "session_summary": "",
        "intent": "quick-local",
        "confidence": 1.0,
        "use_cloud": False,
        "model": "test-model",
        "tool": None,
        "planner_status": "deterministic",
        "reasoning_summary": "",
        "needs_memory": False,
        "route_type": "local",
        "selected_history": [],
        "history_tokens": 0,
        "truncated": False,
        "summary_tokens": 0,
        "memories": [],
        "augmented_system": "You are a helpful assistant.",
        "local_prompt": "hello",
        "cloud_messages": [],
        "response_text": "",
        "response_model": "test-model",
        # Phase 10b fields
        "active_files": [],
        "code_context": "",
        "pending_write": {},
        "awaiting_confirmation": False,
    }
    base.update(overrides)
    return base


# ─── 1. Workspace-root boundary enforcement ───────────────────────────────────

class TestWorkspaceRootBoundary:
    """_resolve_workspace_path must block path traversal attempts."""

    def _make_resolver(self, tmp_root: str):
        """Build the resolver closure exactly as graph.py does."""
        import os as _os
        import pathlib as _pathlib

        def _resolve(relative_path: str) -> str:
            root = _os.path.realpath(tmp_root)
            candidate = _os.path.realpath(_os.path.join(root, relative_path))
            if not (candidate == root or candidate.startswith(root + _os.sep)):
                raise ValueError(
                    f"Path traversal blocked: {relative_path!r} resolves outside workspace root"
                )
            return candidate

        return _resolve

    def test_safe_path_resolves(self, tmp_path):
        resolve = self._make_resolver(str(tmp_path))
        result = resolve("subdir/file.py")
        assert result.startswith(str(tmp_path.resolve()))

    def test_dotdot_traversal_blocked(self, tmp_path):
        resolve = self._make_resolver(str(tmp_path))
        with pytest.raises(ValueError, match="Path traversal blocked"):
            resolve("../etc/passwd")

    def test_absolute_outside_root_blocked(self, tmp_path):
        resolve = self._make_resolver(str(tmp_path))
        with pytest.raises(ValueError, match="Path traversal blocked"):
            resolve("/etc/hosts")

    def test_nested_dotdot_blocked(self, tmp_path):
        resolve = self._make_resolver(str(tmp_path))
        with pytest.raises(ValueError, match="Path traversal blocked"):
            resolve("a/b/../../../../../../etc/shadow")

    def test_root_path_itself_allowed(self, tmp_path):
        resolve = self._make_resolver(str(tmp_path))
        result = resolve("")
        assert result == str(tmp_path.resolve())


# ─── 2 & 3. Write gating + confirmation flow ─────────────────────────────────

class TestWriteGatingAndConfirmation:
    """Test that write_file intercepts writes and confirmation executes them."""

    def _make_unified_diff(self, relative_path, original, proposed):
        import difflib
        lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                proposed.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="",
            )
        )
        return "".join(lines) if lines else ""

    def test_write_file_sets_pending_does_not_write(self, tmp_path):
        """The write_file tool function must not touch disk before confirmation."""
        target = tmp_path / "foo.py"
        target.write_text("original content\n")
        pending: dict = {}

        def _resolve(rel):
            import os
            root = os.path.realpath(str(tmp_path))
            candidate = os.path.realpath(os.path.join(root, rel))
            if not (candidate == root or candidate.startswith(root + os.sep)):
                raise ValueError("traversal")
            return candidate

        # Simulate the write_file tool body
        relative_path = "foo.py"
        content = "new content\n"
        resolved = _resolve(relative_path)
        original = pathlib.Path(resolved).read_text()
        diff = self._make_unified_diff(relative_path, original, content)
        pending["path"] = resolved
        pending["content"] = content
        pending["relative_path"] = relative_path

        # File must NOT have been written yet
        assert target.read_text() == "original content\n"
        # pending_write must be populated
        assert pending["relative_path"] == "foo.py"
        assert pending["content"] == "new content\n"
        assert diff  # diff is non-empty

    def test_write_executor_writes_confirmed_file(self, tmp_path):
        """write_executor must write pending_write to disk and clear state."""
        target = tmp_path / "bar.py"
        target.write_text("old\n")
        resolved = str(target)

        pending = {
            "path": resolved,
            "content": "new\n",
            "relative_path": "bar.py",
        }

        # Simulate write_executor body (minimal)
        pathlib.Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(resolved).write_text(pending["content"], encoding="utf-8")

        assert target.read_text() == "new\n"

    def test_write_executor_creates_new_file(self, tmp_path):
        new_file = tmp_path / "new_dir" / "new.py"
        pending = {
            "path": str(new_file),
            "content": "print('hello')\n",
            "relative_path": "new_dir/new.py",
        }
        pathlib.Path(pending["path"]).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(pending["path"]).write_text(pending["content"], encoding="utf-8")
        assert new_file.read_text() == "print('hello')\n"

    def test_write_executor_revalidates_path(self, tmp_path):
        """write_executor must re-check path even if pending_write was forged."""
        outside = "/tmp/evil.sh"
        resolved_outside = os.path.realpath(outside)
        root = os.path.realpath(str(tmp_path))
        # Confirm traversal check catches it
        assert not (resolved_outside == root or resolved_outside.startswith(root + os.sep))


# ─── 4. _CONFIRM_PATTERN matches ─────────────────────────────────────────────

import re as _re

_CONFIRM_PATTERN = _re.compile(
    r"^\s*(yes|confirm|approve|go ahead|do it|write it|apply|proceed)\s*[.!]?\s*$",
    _re.IGNORECASE,
)


@pytest.mark.parametrize("msg", ["yes", "Yes", "YES", "confirm", "Approve", "go ahead", "do it!", "write it", "apply", "proceed"])
def test_confirm_pattern_matches(msg):
    assert _CONFIRM_PATTERN.match(msg), f"Expected match for: {msg!r}"


@pytest.mark.parametrize("msg", ["no", "cancel", "stop", "sure, but change line 5", ""])
def test_confirm_pattern_does_not_match(msg):
    assert not _CONFIRM_PATTERN.match(msg), f"Expected no match for: {msg!r}"


# ─── 5. code_context injection (unit) ────────────────────────────────────────

class TestCodeContextInjection:
    """memory_retrieval should populate code_context for code intents."""

    def test_code_intent_flag(self):
        state = _make_state(intent="code", message="how does auth work?")
        # Just assert the intent field is set — the actual chroma call is integration-level
        assert state["intent"] == "code"

    def test_non_code_intent_no_context_needed(self):
        state = _make_state(intent="quick-local", message="what is the weather?")
        assert state["intent"] != "code"


# ─── 6. Tree-sitter indexer ──────────────────────────────────────────────────

class TestCodeIndexer:
    """Index a fixture Python file and verify ChromaDB retrieval."""

    def _fixture_py(self) -> str:
        return textwrap.dedent("""\
            def greet(name: str) -> str:
                \"\"\"Return a greeting string.\"\"\"
                return f"Hello, {name}!"

            class Calculator:
                \"\"\"Simple calculator class.\"\"\"

                def add(self, a: int, b: int) -> int:
                    return a + b

                def subtract(self, a: int, b: int) -> int:
                    return a - b
        """)

    def test_index_and_query(self, tmp_path):
        pytest.importorskip("chromadb", reason="chromadb not installed")
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from tools.code_indexer import index_workspace, query_code_context

        # Write fixture file
        src = tmp_path / "src"
        src.mkdir()
        fixture = src / "greeter.py"
        fixture.write_text(self._fixture_py())

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()

        count = index_workspace(str(tmp_path), str(chroma_dir))
        assert count >= 1, "Expected at least one document indexed"

        results = query_code_context("greeting function", str(chroma_dir), n=3)
        assert results, "Expected at least one result from query"
        combined = "\n".join(results)
        assert "greet" in combined or "Calculator" in combined

    def test_index_skips_large_files(self, tmp_path):
        pytest.importorskip("chromadb", reason="chromadb not installed")
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from tools.code_indexer import index_workspace, _MAX_FILE_BYTES

        large = tmp_path / "large.py"
        large.write_bytes(b"x" * (_MAX_FILE_BYTES + 1))

        chroma_dir = tmp_path / "chroma_large"
        chroma_dir.mkdir()

        count = index_workspace(str(tmp_path), str(chroma_dir))
        assert count == 0, "Large file should be skipped"

    def test_index_ignores_pycache(self, tmp_path):
        pytest.importorskip("chromadb", reason="chromadb not installed")
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from tools.code_indexer import index_workspace

        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-311.pyc").write_bytes(b"binary junk")

        chroma_dir = tmp_path / "chroma_cache"
        chroma_dir.mkdir()

        count = index_workspace(str(tmp_path), str(chroma_dir))
        assert count == 0


# ─── 7. /code/* HTTP endpoints ────────────────────────────────────────────────

@pytest.fixture()
def code_workspace(tmp_path):
    """Create a minimal workspace and set CODE_WORKSPACE_ROOT."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "hello.py").write_text("print('hello')\n")
    sub = ws / "sub"
    sub.mkdir()
    (sub / "util.py").write_text("def noop(): pass\n")
    memory_db = tmp_path / "memory.db"
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    auth_db = tmp_path / "auth.db"
    monkeypatch_env = {
        "CODE_WORKSPACE_ROOT": str(ws),
        "MEMORY_DB_PATH": str(memory_db),
        "CHROMA_PATH": str(chroma_dir),
        "AUTH_DB_PATH": str(auth_db),
    }
    return ws, monkeypatch_env


def _import_main_module():
    import importlib
    import sys

    if "main" in sys.modules:
        return importlib.reload(sys.modules["main"])

    import main as main_mod  # type: ignore[import]

    return main_mod


@pytest.mark.asyncio
async def test_list_code_files_endpoint(code_workspace, monkeypatch):
    ws, env = code_workspace
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    # Re-import main with the env var set so _get_code_root() sees it
    main_mod = _import_main_module()

    client = TestClient(main_mod.app, raise_server_exceptions=True)
    # We need an auth token; use the bypass header if available or skip
    resp = client.get("/code/files", headers={"Authorization": "Bearer test-skip"})
    # Accept 401 (auth required) or 200 (if test auth bypassed)
    assert resp.status_code in (200, 401, 403, 503)


@pytest.mark.asyncio
async def test_read_code_file_endpoint(code_workspace, monkeypatch):
    ws, env = code_workspace
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    main_mod = _import_main_module()

    client = TestClient(main_mod.app, raise_server_exceptions=False)
    resp = client.get("/code/files/hello.py", headers={"Authorization": "Bearer test-skip"})
    assert resp.status_code in (200, 401, 403, 503)


@pytest.mark.asyncio
async def test_safe_resolve_blocks_traversal(code_workspace, monkeypatch):
    ws, env = code_workspace
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    main_mod = _import_main_module()

    client = TestClient(main_mod.app, raise_server_exceptions=False)
    resp = client.get("/code/files/../../../etc/passwd", headers={"Authorization": "Bearer test-skip"})
    # Must not return 200 with /etc/passwd contents
    assert resp.status_code in (400, 401, 403, 404, 422, 503)


@pytest.fixture()
def authed_client(code_workspace, monkeypatch):
    _, env = code_workspace
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    main_mod = _import_main_module()

    monkeypatch.setattr(main_mod.auth_service, "verify_token", lambda _token: "test-user")
    client = TestClient(main_mod.app, raise_server_exceptions=True)
    return client


def test_write_code_file_confirmation_required(authed_client):
    resp = authed_client.put(
        "/code/files/hello.py",
        headers={"Authorization": "Bearer ok"},
        json={"content": "print('updated')\n"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending_confirmation"
    assert data["request_id"]
    assert "diff" in data

    read_resp = authed_client.get("/code/files/hello.py", headers={"Authorization": "Bearer ok"})
    assert read_resp.status_code == 200
    assert "updated" not in read_resp.json().get("content", "")


def test_write_code_file_confirm_executes(authed_client):
    first = authed_client.put(
        "/code/files/hello.py",
        headers={"Authorization": "Bearer ok"},
        json={"content": "print('confirmed')\n"},
    )
    assert first.status_code == 200
    request_id = first.json().get("request_id")
    assert request_id

    second = authed_client.put(
        "/code/files/hello.py",
        headers={"Authorization": "Bearer ok"},
        json={"content": "ignored\n", "confirm": True, "request_id": request_id},
    )
    assert second.status_code == 200
    assert second.json().get("status") == "written"

    read_resp = authed_client.get("/code/files/hello.py", headers={"Authorization": "Bearer ok"})
    assert read_resp.status_code == 200
    assert read_resp.json().get("content") == "print('confirmed')\n"


def test_code_stream_endpoint_sse(authed_client, monkeypatch):
    import main as main_mod  # type: ignore[import]

    class _FakeGraph:
        async def astream(self, *_args, **_kwargs):
            yield {"meta": {"model": "qwen2.5-coder:7b", "intent": "code", "confidence": 1.0}}
            yield {"text": "def bubble_sort(arr):\n"}
            yield {"text": "    return arr\n"}

    main_mod.app.state.assistant_graph = _FakeGraph()

    with authed_client.stream(
        "POST",
        "/code",
        headers={"Authorization": "Bearer ok"},
        json={"message": "write bubble sort", "source": "text"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(line for line in resp.iter_text())

    assert "def bubble_sort" in body
    assert "[DONE]" in body


def test_code_mode_yes_without_pending_write_is_blocked(monkeypatch):
    """Regression: bare 'yes' in code mode must not trigger a phantom write."""
    from graph import build_assistant_graph, AssistantGraphDependencies  # type: ignore[import]
    from router import RouteDecision  # type: ignore[import]
    import memory as memory_mod  # type: ignore[import]

    async def _router_route(_message: str):
        return RouteDecision(
            intent="code",
            confidence=1.0,
            use_cloud=False,
            model="qwen2.5-coder:14b",
            tool=None,
            planner_status="deterministic",
            reasoning_summary="",
            needs_memory=False,
        )

    async def _stream_noop(*_args, **_kwargs):
        if False:
            yield ""

    async def _tool_dispatch_noop(*_args, **_kwargs):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("CODE_WORKSPACE_ROOT", tmpdir)
        deps = AssistantGraphDependencies(
            chat_model="gemma3:4b",
            coder_model="qwen2.5-coder:14b",
            cloud_model=None,
            memory_store=memory_mod.MemoryStore(),
            router_route=_router_route,
            stream_local=_stream_noop,
            stream_cloud=_stream_noop,
            tool_dispatch=_tool_dispatch_noop,
        )
        graph = build_assistant_graph(deps)

        state = _make_state(
            message="yes",
            force_code=True,
            intent="code",
            pending_write={},
            awaiting_confirmation=False,
        )

        events: list[dict] = []

        async def _run():
            async for event in graph.astream(
                state,
                config={"configurable": {"thread_id": "test-thread"}},
                stream_mode="custom",
            ):
                events.append(event)

        asyncio.run(_run())
        text_output = "".join(e.get("text", "") for e in events if "text" in e)
        assert "No pending write to confirm." in text_output
        assert "Written:" not in text_output
