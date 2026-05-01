# Security and Correctness Review Checklist

Use this checklist for every pull request and release gate.

## Severity Tiers

- Tier 0 (critical): auth/session boundaries, file write paths, workspace-root enforcement, cross-user memory isolation, graph routing that can trigger tools.
- Tier 1 (high): external adapters, model fallback and cloud routing behavior, container/network exposure.
- Tier 2 (medium): UX state machines, maintainability and documentation drift.

## Per-PR Required Checks

1. Run local baseline checks:
   - Recommended fast precheck: `bash scripts/review_changed_tests.sh --base origin/main`
   - Optional local-only mode: `bash scripts/review_changed_tests.sh --base origin/main --allow-known-failures`
   - Local deselection file: `docs/review/KNOWN_FAILURES.txt`.
   - `bash scripts/review_baseline.sh`
   - If `gitleaks` is installed locally, the script runs a secret scan.
   - If `gitleaks` is not installed locally, CI still enforces secret scanning.
   - If fast precheck fails, treat failures as real regressions unless proven unrelated.
2. CI fast feedback:
   - Pull requests run a changed-files test job (`changed-files-fast-tests`).
   - This job is currently non-blocking (`continue-on-error`) while known regressions are being burned down.
3. Confirm CI `Review Gates` workflow passes.
4. For Tier 0 and Tier 1 changes, require explicit reviewer signoff from subsystem owner.

## Security Checklist

- Auth is enforced on protected endpoints and no new bypass route was introduced.
- Cookie and token handling still follows secure defaults and does not leak into logs.
- Any new file I/O path is validated against workspace-root traversal protections.
- Write operations remain confirmation-gated where required.
- Error responses use the standardized shape: `{ "error": str, "code": str, "retryable": bool }`.
- New logging statements do not include credentials, tokens, secrets, or PII.
- External API/tool integrations include timeout, retry/error handling, and safe fallback behavior.

## Correctness Checklist

- Graph state fields remain backward-compatible and checkpoint resume still works.
- Routing decisions preserve deterministic fast paths (such as explicit music commands).
- Memory retrieval and mutation remain user-scoped with no cross-user leakage.
- Tool contracts remain stable (`async def run(params: dict) -> dict`) and response shapes are unchanged or migrated deliberately.
- Tests for changed behavior are added or updated before merge.

## Modularity and Maintainability Checklist

- New logic is placed in the appropriate module (`backend/tools/`, graph node, or endpoint layer).
- No duplicated business logic across endpoint handler and graph node unless intentionally mirrored.
- Environment variables are used for configuration, without hardcoding model or host settings.
- Public APIs and data contracts are documented when changed.

## Release Gate Requirements

- No open critical findings.
- No open high findings without documented risk acceptance.
- Tier 0 tests pass.
- Security scan findings are triaged and resolved or explicitly waived.
- Regression notes are captured in a release gate document.
