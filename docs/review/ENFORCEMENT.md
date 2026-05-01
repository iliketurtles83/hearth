# Review Enforcement Setup

This document explains how to enforce the review controls introduced in this repository.

## 1. Enable Required Reviews (GitHub Branch Protection)

For the protected branch (usually `main`):

1. Go to repository settings -> Branches.
2. Create or edit a branch protection rule for `main`.
3. Enable:
   - Require a pull request before merging
   - Require approvals (recommended: at least 1)
   - Require review from Code Owners
   - Dismiss stale pull request approvals when new commits are pushed

This makes `CODEOWNERS` assignments mandatory for covered paths.

## 2. Require CI Status Checks

In the same branch protection rule, enable required status checks and select:

- `backend-review-gates`

Optional early signal:

- `changed-files-fast-tests` (non-blocking for now; useful for quick PR feedback)

The `backend-review-gates` workflow includes:

- secret scanning (`gitleaks` action)
- focused regression tests
- dependency vulnerability audit
- static security scan (`bandit`, medium/high threshold)

The `changed-files-fast-tests` job runs `scripts/review_changed_tests.sh --base origin/main` on pull requests.

## 3. Keep Ownership Mapping Current

Update `.github/CODEOWNERS` whenever critical files move or new Tier 0 surfaces are added.

Recommended critical mappings include:

- `backend/main.py`
- `backend/auth.py`
- `backend/graph.py`
- `backend/memory.py`
- code tool and related tests
- deployment and edge config (`docker-compose.yml`, `backend/Dockerfile`, `caddy/Caddyfile`)

## 4. PR Workflow Expectations

Contributors should:

1. Fill in `.github/pull_request_template.md`.
2. Run `bash scripts/review_changed_tests.sh --base origin/main` for quick feedback.
3. Run `bash scripts/review_baseline.sh` before requesting review.
4. Call out Tier 0/Tier 1 risk areas explicitly in the PR summary.

## 5. Triage Policy

- Block merge for: critical/high findings unless waived with explicit risk acceptance.
- Medium findings: fix promptly; allow temporary waiver only with owner signoff and follow-up issue.
- Low findings: track in backlog and batch-fix regularly.
