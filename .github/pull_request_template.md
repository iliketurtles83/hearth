## Summary

- What changed:
- Why it changed:
- Risk level (Tier 0 / Tier 1 / Tier 2):

## Validation

- [ ] Ran fast changed-files check: `bash scripts/review_changed_tests.sh --base origin/main`
- [ ] If using local deselection list, noted it explicitly: `--allow-known-failures`
- [ ] Ran local baseline checks: `bash scripts/review_baseline.sh`
- [ ] Added/updated tests for changed behavior
- [ ] Verified no regression in affected endpoints or graph/tool flows

## Security and Privacy

- [ ] No auth bypass introduced for protected routes
- [ ] No path traversal risk introduced in file I/O paths
- [ ] No secrets/tokens/PII added to logs
- [ ] Error responses follow standardized shape where applicable
- [ ] Cloud fallback behavior remains explicit and non-silent

## Correctness and Modularity

- [ ] State/schema changes are checkpoint-compatible (if applicable)
- [ ] Tool contract compatibility preserved (`async def run(params: dict) -> dict`)
- [ ] Deterministic fast paths unchanged unless explicitly intended
- [ ] Configuration remains env-driven (no hardcoded runtime hosts/models)

## Review Notes

- Critical files touched:
- Follow-up tasks (if any):
