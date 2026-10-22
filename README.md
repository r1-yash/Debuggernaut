# Debuggernaut

An autonomous Python bug-fixing agent for GitHub repositories. Given a benchmark configuration, it clones each repository, fetches its selected issue, uses an LLM to explore the relevant files and propose a full-file fix, runs the configured test command locally, diagnoses and retries failed fixes, and can optionally open a pull request for a verified fix.

The agent explores repositories through targeted read, directory-listing, and code-search tools instead of adding the whole repository to the model context.

## Architecture

```
Benchmark JSON
    ↓
Clone repository at pinned commit + fetch GitHub issue
    ↓
LangGraph resolution loop
    ↓
LLM exploration tools: read file, list directory, search code
    ↓
Structured fix proposal → validated against the repository
    ↓
Apply fix + run configured test command (120s timeout)
    ↓
Pass ─────────────────→ optional fork, commit, push, and pull request
    ↓ fail
LLM diagnosis → retry (up to three attempts)
```

## Components

- **LangGraph** — agent orchestration and state (`swe_agent/agent/loop.py`)
- **Provider abstraction** — Gemini (default) or Qwen via OpenRouter, swappable through `LLM_PROVIDER` (`swe_agent/llm.py`)
- **Exploration tools** — `repository_read_file`, `repository_list_directory`, `repository_search_code`
- **Fix executor** — applies the proposed full-file rewrite, runs the configured test command, and reverts on failure (`swe_agent/agent/executor.py`)
- **Diagnosis** — classifies each failed attempt as `wrong_file` or `wrong_fix`; wrong-file targets are excluded from later attempts
- **GitHub API** — issue listing, fork, branch, commit, push, and PR creation
- **Baseline validation** — `--validate` mode confirms each entry's pinned commit reproduces a test failure before any LLM call (`swe_agent/benchmark/validate.py`)
- **Streamlit** — issue selection + live execution trace (`streamlit_app.py`)
- **JSON run reports** — per-entry results written to `runs/`

The entire repository is never dumped into LLM context — the agent selectively explores only what's relevant to the selected issue.

## Setup

Requires Python 3.13+ and `git`.

```bash
uv sync                      # or: pip install -r requirements.txt
cp .env.example .env         # then fill in your API key
```

`.env` selects the provider:

```dotenv
# Gemini (default)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=<your-key>

# Or Qwen via OpenRouter
# LLM_PROVIDER=qwen
# LLM_MODEL=qwen/qwen3-coder:free
# QWEN_API_KEY=<your-OpenRouter-key>
# QWEN_BASE_URL=https://openrouter.ai/api/v1
```

Opening a pull request additionally requires `GITHUB_TOKEN` with permission to fork the target repository and open a PR against it.

## Usage

### CLI benchmark

Benchmark configs are JSON objects mapping `owner/repository` to an issue, the pinned commit to check out, and a test command (see `benchmark/repos.json`):

```json
{
  "r1-yash/toy-bug-repo": {
    "issue_number": 1,
    "commit": "REPLACE_WITH_BUGGY_COMMIT_SHA",
    "test_command": ["python", "-m", "pytest", "test_calc.py"]
  }
}
```

`commit` must be the full 40-character hexadecimal SHA of the commit **before** the fix, so the bug actually exists in the checked-out tree — a clone at HEAD may already contain the upstream fix. The shipped toy entry uses a placeholder that config loading rejects until you replace it with the real pre-fix SHA.

An entry may also include an optional `test_patch`: a path to a unified diff, relative to the config file (for example `benchmark/patches/<owner>__<repo>__<issue>.diff`):

```json
"owner/repository": {
  "issue_number": 42,
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "test_command": ["python", "-m", "pytest", "tests/test_fix.py"],
  "test_patch": "patches/owner__repository__42.diff"
}
```

The patch is applied immediately after the pinned commit is checked out and is committed locally as a baseline, so agent edits, per-attempt diffs, and git-based scope checks are measured against the post-patch tree. Because the clone's HEAD moves to that local baseline commit, git-derived versions such as setuptools-scm may differ from the pinned commit. The agent may read the patched test files (reading is allowed) but cannot modify them; entries measure "issue + failing test", not hidden-test resolution.

```bash
uv run main.py --config benchmark/repos.json --validate  # check entries reproduce (no LLM calls)
uv run main.py --config benchmark/repos.json            # fix only
uv run main.py --config benchmark/repos.json --open-pr  # also open a PR on success
```

`--validate` checks each entry *before* any model is called: it checks out the pinned commit, applies the entry's optional test patch, runs the entry's test command, and confirms the tests fail (exit code 1 for pytest). Entries whose test patch cannot be applied are reported as `test_patch_failed`; entries whose tests already pass, cannot be run (timeout, command not found, or pytest exit codes other than 0/1), or whose commit cannot be fetched are reported as not reproducible. Each entry's exit code and the last ~10 lines of test output are printed. It makes zero LLM calls and needs no API key; the process exits non-zero if any entry is not reproducible.

### Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

Enter a repository URL, fetch its open issues, choose one, provide a test command, and run the agent. Each attempt's changed file, reasoning, and test output are shown live.

### Tests

```bash
uv run pytest
```

## Run reports

Every benchmark entry writes a JSON report to `runs/` recording repository, issue, provider, model, status, attempt count, per-call latency and token usage, and the PR URL when one was created. Reports also record whether a benchmark test patch was applied and the paths it touched (never its contents). Prompts, repository contents, and API keys are never written to reports. When a provider omits usage metadata, token values are `null` — they are never estimated.

## Guardrails

- **Path confinement** — every file read, write, or listing resolves through `resolve_repository_path`, which rejects any path escaping the cloned repository's root (`swe_agent/agent/paths.py`)
- **Proposal validation** — a fix is rejected before application if the LLM's proposed `file_path` does not exist in the repository, preventing hallucinated or stale paths from crashing the run
- **Protected test files** — the agent cannot modify files under `tests/`/`test/`, `test_*.py`, `*_test.py`, `conftest.py`, files named by the entry's test command, or files touched by a benchmark test patch (recorded as guard-event kind `test_patch_file`); reading them is allowed
- **Structured output** — fixes and diagnoses are returned as schema-validated JSON, not free text
- **Revert on failure** — a fix that fails its test command is restored to its original contents
- **Bounded execution** — test commands are capped at 120 seconds and the resolution loop at three attempts, so a single issue cannot run indefinitely
- **Least-scope commits** — PR creation validates that only the fixed file is modified before staging, committing, and pushing to a dedicated `swe-agent/fix-issue-<n>` branch on a fork

## Project layout

```
main.py                    CLI benchmark entry point
streamlit_app.py           Streamlit UI
benchmark/repos.json       Benchmark configuration
src/swe_agent/
  llm.py                   Provider-neutral clients + token usage tracking
  agent/
    loop.py                LangGraph propose → test → diagnose loop
    fixer.py               Exploration tools + fix proposal
    executor.py            Apply, test, and revert fixes
    diagnose.py            Failure classification
    pr.py                  Fork, commit, push, and PR creation
    paths.py               Path-traversal guardrail
  ingestion/               Repository cloning and GitHub issue fetching
  benchmark/               Config loading, baseline validation, runner, and report writing
tests/                     pytest suite
runs/                      Per-entry JSON run reports
workspace/                 Cloned target repositories
```

## Qwen evaluation

Debuggernaut supports Gemini and Qwen through one provider interface. Qwen is configured through [OpenRouter](https://openrouter.ai/), which provides an OpenAI-compatible API:

```dotenv
LLM_PROVIDER=qwen
LLM_MODEL=qwen/qwen3-coder:free
QWEN_API_KEY=<your-OpenRouter-key>
# Optional override:
QWEN_BASE_URL=https://openrouter.ai/api/v1
```

OpenRouter documents `https://openrouter.ai/api/v1` as an OpenAI SDK drop-in base URL. Obtain an API key from OpenRouter and check the current model catalog, availability, and free-tier conditions before running an evaluation. [OpenRouter quickstart](https://openrouter.ai/docs/quickstart)

## Scope

The current implementation targets Python repository issues supplied in benchmark JSON. It is a CLI and Streamlit workflow — there is no server backend.

**Investigate → propose → test → diagnose → retry → optionally open a PR**
