# Debuggernaut

An autonomous Python bug-fixing agent for GitHub repositories. Given a benchmark configuration, it clones each repository at a pinned commit, fetches its selected issue, builds an isolated test environment, uses an LLM to explore the relevant files and propose a full-file fix, runs the configured test command, diagnoses and retries failed fixes, and records a machine-readable result for every entry. A verified fix can optionally be turned into a pull request outside benchmark mode.

The agent explores repositories through targeted read, directory-listing, and code-search tools instead of adding the whole repository to the model context, and every run is gated by a baseline check that confirms the pinned tests actually reproduce the bug before a single token is spent.

**Investigate → propose → test → diagnose → retry**

## How it works

```
Benchmark JSON (owner/repository → issue, pinned SHA, test command, optional test patch)
    ↓
Sweep stale workspaces → clone repository at pinned commit
    ↓
Create isolated environment: uv venv, throwaway HOME/TMP, allowlisted env, install plan
    ↓
Apply optional test patch and commit it locally as the baseline
    ↓
Baseline gate: run the configured tests, confirm they fail (no LLM calls)
    ↓
Fetch the GitHub issue from the API
    ↓
LangGraph resolution loop (max 3 attempts)
    ↓
LLM exploration tools: read file, list directory, search code
    ↓
Structured fix proposal → validated against the repository
    ↓
Apply fix + run the test command inside the isolated environment
    ↓
Pass ─────────────────→ record success (optional PR outside benchmark mode)
    ↓ fail
Revert fix → LLM diagnosis (wrong_file | wrong_fix) → retry
    ↓
Close environment (removes venv and clone) → write JSON report + run summary
```

## Components

- **LangGraph** — agent orchestration and state (`swe_agent/agent/loop.py`)
- **Provider abstraction** — Gemini (default), Qwen via OpenRouter, or Token Harbor through one interface, selected with `LLM_PROVIDER` or `--provider` (`swe_agent/llm.py`)
- **Exploration tools** — `repository_read_file`, `repository_list_directory`, `repository_search_code`
- **Fix executor** — applies the proposed full-file rewrite, runs the configured test command, and reverts on failure (`swe_agent/agent/executor.py`)
- **Diagnosis** — classifies each failed attempt as `wrong_file` or `wrong_fix`; wrong-file targets are excluded from later attempts
- **Isolated environments** — one `uv` virtual environment and throwaway `HOME`/`TMP` per entry, with a deterministic install plan and per-entry deadlines (`swe_agent/benchmark/environment.py`)
- **Baseline gate** — every run and every `--validate` entry reproduces the pinned failure before the agent starts, and classifies infrastructure failures separately from agent failures
- **GitHub API** — issue listing, fork, branch, commit, push, and PR creation
- **Streamlit** — issue selection + live execution trace (`streamlit_app.py`)
- **JSON run reports** — per-entry results and a benchmark summary written to `runs/`
- **Benchmark tooling** — candidate generation from SWE-bench-extra and reproducible regeneration of the committed benchmark set (`benchmark/tools/`)

The entire repository is never dumped into LLM context — the agent selectively explores only what's relevant to the selected issue.

## Requirements

- Python 3.13+ and `git` for the harness itself
- [`uv`](https://docs.astral.sh/uv/) on `PATH` for benchmark execution, because each target repository is installed into its own managed virtual environment
- A one-time managed Python for target environments:

```bash
uv python install 3.11
```

Python download time is excluded from per-entry setup deadlines, and managed interpreters and package caches are shared under `workspace/` so a throwaway `HOME` does not force a re-download for every entry.

## Setup

```bash
uv sync                      # supported path
cp .env.example .env         # then fill in your API key
```

`requirements.txt` is a partial fallback and omits `openai`, which the Qwen and Token Harbor providers require; `uv sync` installs the full set from `pyproject.toml`.

## Providers

Gemini is the default. All three providers are reachable through the same interface, and the CLI flags (`--provider`, `--model`, `--base-url`) override the environment.

| Provider | `LLM_PROVIDER` | Default model | Required key | Default base URL |
| --- | --- | --- | --- | --- |
| Gemini | `gemini` | `gemini-3.5-flash-lite` | `GOOGLE_API_KEY` | Google SDK default |
| Qwen (OpenRouter) | `qwen` | `qwen/qwen3-coder:free` | `QWEN_API_KEY` | `https://openrouter.ai/api/v1` |
| Token Harbor | `tokenharbor` | `deepseek-v4.1-flash:free` | `TOKENHARBOR_API_KEY` | `https://tokenharbor.ai/v1` |

Model resolution order is `--model`, then `LLM_MODEL`, then `<PROVIDER>_MODEL`, then the provider default. The base URL resolves from `--base-url`, then `<PROVIDER>_BASE_URL`, then the provider default.

```dotenv
# Gemini (default)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=<your-key>

# Qwen via OpenRouter
# LLM_PROVIDER=qwen
# LLM_MODEL=qwen/qwen3-coder:free
# QWEN_API_KEY=<your-OpenRouter-key>
# QWEN_BASE_URL=https://openrouter.ai/api/v1

# Token Harbor (requires an explicit input-token limit)
# LLM_PROVIDER=tokenharbor
# TOKENHARBOR_API_KEY=<your-Token-Harbor-key>
# TOKENHARBOR_MAX_INPUT_TOKENS=60000
```

Opening a pull request additionally requires `GITHUB_TOKEN` with permission to fork the target repository and open a PR against it. Setting `GITHUB_TOKEN` also raises GitHub API rate limits for issue fetching.

### Token limits and retries

- `--max-input-tokens` sets the request-input ceiling. Every Token Harbor request is counted locally with the official DeepSeek V4.1 prompt encoding (see [Input token counting](#input-token-counting)); an oversized request is refused as `context_too_large` without contacting the provider. Token Harbor requires this value, either as a flag or through `TOKENHARBOR_MAX_INPUT_TOKENS`. Other providers keep the dependency-free conservative estimate (JSON payload size / 2).
- `--max-retries` (default `6`, `<PROVIDER>_MAX_RETRIES`) bounds retries after rate limits, server errors, and connection failures. Retries honor `Retry-After`; `--max-retry-after` (default 60 s) caps the accepted wait, and a longer advertised wait becomes `provider_unavailable` rather than a silent stall.
- Every OpenAI-compatible provider runs a preflight before the first entry is scheduled: one function tool with `tool_choice="auto"`, matching how the agent really calls the model. A rejected key, unknown model, wrong base URL, timeout, or a reply with no tool call fails the run with a cause-specific error and exit code `3`, before any clone or report is written.
- API keys are redacted from surfaced provider errors, and benchmark execution never forwards parent secrets into target-repository processes.

#### Input token counting

`src/swe_agent/token_count.py` counts the complete outgoing request before it is sent. For Token Harbor's `deepseek-v4.1-flash:free`, `deepseek-recipe` (declared in `pyproject.toml`) converts the OpenAI-compatible body into a DeepSeek V4.1 conversation and renders it with the model's own prompt template, and `tokenizer.json` from `deepseek-ai/DeepSeek-V4.1-Flash` (about 6 MB, tokenizer only — never model weights) tokenizes that prompt. Messages, tool declarations, tool calls, tool results, control tokens, message boundaries, and thinking-mode formatting are all counted, so the limit is checked against the real prompt instead of a byte-length guess.

The tokenizer is loaded once per process, from `DEEPSEEK_TOKENIZER_PATH` when set (offline, pinned file) or downloaded and cached from `DEEPSEEK_TOKENIZER_MODEL` (`deepseek-ai/DeepSeek-V4.1-Flash` by default). If `deepseek-recipe` is missing, the tokenizer cannot be loaded, or the reference conversion rejects the body (a `json_object` request whose prompt never says "json", which DeepSeek's API rejects with HTTP 400), the request keeps the conservative byte-length estimate. Every fallback is logged with its reason, and each call records both `input_token_count_method` and `input_token_count_reason` in the run record, so an exact count is never confused with an approximation. Tool-result caps and other agent-facing limits are unchanged; `fixer.py` still derives its 24,000-character tool-result budget from `max_input_tokens` with the same 2-characters-per-token rule.

OpenRouter documents `https://openrouter.ai/api/v1` as an OpenAI SDK drop-in base URL. Obtain an API key from OpenRouter and check the current model catalog, availability, and free-tier conditions before running an evaluation. [OpenRouter quickstart](https://openrouter.ai/docs/quickstart)

## Usage

### Benchmark config

Benchmark configs are JSON objects mapping `owner/repository` to an issue, the pinned commit to check out, and a test command. For example:

```json
{
  "r1-yash/toy-bug-repo": {
    "issue_number": 1,
    "commit": "REPLACE_WITH_BUGGY_COMMIT_SHA",
    "test_command": ["python", "-m", "pytest", "test_calc.py"]
  }
}
```

`commit` must be the full 40-character hexadecimal SHA of the commit **before** the fix, so the bug actually exists in the checked-out tree — a clone at HEAD may already contain the upstream fix. Config loading rejects placeholder values that are not full SHAs.

An entry may also include an optional `test_patch`: a path to a unified diff, relative to the config file (for example `benchmark/patches/<owner>__<repo>__<issue>.diff`):

```json
"owner/repository": {
  "issue_number": 42,
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "test_command": ["python", "-m", "pytest", "tests/test_fix.py"],
  "test_patch": "patches/owner__repository__42.diff"
}
```

The patch is applied immediately after the pinned commit is checked out and is committed locally as a baseline, so agent edits, per-attempt diffs, and git-based scope checks are measured against the post-patch tree. Patch paths must stay inside the config file's directory and must apply cleanly; there is no three-way or whitespace-fuzzy fallback. Because the clone's HEAD moves to that local baseline commit, git-derived versions such as setuptools-scm may differ from the pinned commit. The agent may read the patched test files (reading is allowed) but cannot modify them; entries measure "issue + failing test", not hidden-test resolution.

Test patches can never be pushed to an external repository: `run_benchmark` refuses to run with pull-request creation enabled when any entry defines a test patch, and the PR layer independently refuses a patched checkout as a second line of defense.

### Validate entries

```bash
uv run main.py --config benchmark/repos.json --validate
uv run main.py --config benchmark/repos.json --validate --limit 5
uv run main.py --config benchmark/repos.json --validate --results-file validate.jsonl
```

`--validate` checks each entry *before* any model is called: it checks out the pinned commit, creates the isolated environment, applies the entry's optional test patch, runs the entry's test command, and confirms the tests fail. It makes zero LLM calls and needs no API key.

Each entry resolves to one of:

| Status | Meaning |
| --- | --- |
| `fails_as_expected` | The pinned tests reproduce the failure |
| `passes_unexpectedly` | Nothing to fix; the tests already pass |
| `could_not_install` | The environment could not be prepared, or a third-party dependency was missing |
| `test_timeout` | The test command exceeded its timeout |
| `test_patch_failed` | The entry's test patch did not apply |
| `could_not_run` | The clone, checkout, or test process could not be started |

A `ModuleNotFoundError` for a module that is not part of the repository's own top-level packages is treated as an installation failure, not as a reproduced bug; a missing internal module counts as an ordinary expected failure.

Progress is streamed one line per entry with its elapsed time. Each entry's status, exit code, and the last ~10 lines of test output are printed at the end. `--limit N` checks only the first N entries in config order, and `--results-file` appends one JSON object per completed entry, flushed as it finishes, so a long validation can be resumed or inspected mid-run. Both options require `--validate`. The process exits non-zero if any entry is not reproducible, and an interrupt prints the partial summary and exits `130`.

### Run the benchmark

```bash
uv run main.py --config benchmark/repos.json
uv run main.py --config benchmark/repos.json --provider tokenharbor --max-input-tokens 60000
uv run main.py --config benchmark/repos.json --retry-errored
```

Each entry goes through the full lifecycle described above: clone, environment setup, optional test patch, baseline gate, issue fetch, resolution loop, and report writing. An entry that fails its baseline is recorded as `errored`, `could_not_install`, or `test_timeout` — never as an agent failure — and the agent is not started.

A run stops scheduling new entries at the first `provider_unavailable`, because the remaining entries would fail for the same reason.

`--open-pr` is not supported in benchmark mode and is rejected while parsing arguments. Pull requests are created through the Streamlit UI or by calling `create_fix_pull_request` directly.

### Timeouts

Execution is bounded at three levels, all settable on both `--validate` and benchmark runs:

| Flag | Default | Scope |
| --- | --- | --- |
| `--install-timeout` | 600 s | Each installation command |
| `--test-timeout` | 300 s (validate) / 120 s (run) | Baseline and agent test commands |
| `--setup-deadline` | 1800 s | Total setup time for one entry |

Test and install processes run in their own process group, so a timeout kills the whole process tree instead of leaving a stray `pytest` or `pip` behind.

### Resume and rerun

Reports in `runs/` are the resume log. On start, entries that already have a terminal report are skipped when the report matches the same repository, issue, commit, `provider`, `model`, and `base_url_host`. Terminal statuses are `success`, `failed`, `test_timeout`, `context_too_large`, `could_not_install`, `provider_unavailable`, and `errored`.

`--retry-errored` re-schedules the infrastructure outcomes (`errored`, `could_not_install`, `provider_unavailable`) so a fixed environment or a restored provider can be retried without repeating successful entries.

### Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

Enter a repository URL, fetch its recent open issues, choose one, provide a test command, and run the agent. Each attempt's changed file, reasoning, and test output are shown live, and a verified fix can optionally open a pull request. The UI runs on the host environment with the Gemini SDK: it does not use the per-entry isolation, the baseline gate, or report writing, which are benchmark-only features.

### Tests

```bash
uv run pytest
```

The suite covers the loop, guardrails, executor, providers, environments, baseline classification, validation, reporting, cloning, issue fetching, and the benchmark tooling. Test collection is limited to `tests/`.

## Benchmark suite

The committed benchmark set is generated from `benchmark/final_ids.txt` by `python benchmark/tools/finalize_benchmark.py`; its test patches live under `benchmark/patches/` and the resulting config is `benchmark/repos.json`.

`benchmark/tools/` contains standalone tooling that nothing under `src/swe_agent` imports:

- `build_candidates.py` — turns a pinned `nebius/SWE-bench-extra` revision into candidate entries: shape filtering, GitHub closing-issue resolution, heavy-repository denylisting, and JSON/Markdown reports. Gold fixes are written for local inspection only and are never referenced by the config.
- `issue_resolution.py` — resolves the issue closed by a pull request, preferring GraphQL `closingIssuesReferences` and falling back to closing keywords in the PR body, with an on-disk cache.
- `finalize_benchmark.py` — regenerates the committed config and the copied test patches from the selected ids, rejecting missing or duplicate ids, unexpected output files, and patches that touch anything other than test files.
- `heavy_repos.txt` — repositories whose test environments need heavy or system-level dependencies.

The builder only performs metadata and shape screening. The correctness gate is `--validate`, which checks out each pinned commit, applies the test patch, and confirms the tests reproduce the failure.

```bash
python -m pip install -r benchmark/tools/requirements.txt
export GITHUB_TOKEN=...   # recommended; GraphQL needs it
python benchmark/tools/build_candidates.py --dry-run
python benchmark/tools/finalize_benchmark.py
```

Candidate output is written to `benchmark/candidates/`, which is gitignored.

## Execution model

### Isolated environments

Every benchmark entry gets its own `uv` virtual environment under `workspace/.environments/<owner>__<repository>__<issue>/`, together with a throwaway `HOME` and temporary directory. The child environment is rebuilt from an explicit allowlist (`PATH`, locale, `TZ`, and the Windows system variables); `HOME`, `TMPDIR`, `TEMP`, and `TMP` are replaced, and `CI`, `PYTHONNOUSERSITE`, `PYTHONDONTWRITEBYTECODE`, and `GIT_TERMINAL_PROMPT=0` are set. Parent secrets such as `GITHUB_TOKEN`, `GOOGLE_API_KEY`, and `QWEN_API_KEY` are never forwarded. Managed interpreters (`workspace/.uv-python`) and package caches (`workspace/.cache/uv`, `workspace/.cache/pip`) are shared so cleanup stays cheap.

Installation runs a deterministic plan:

1. `pytest` into the entry's environment (a failure here aborts setup)
2. Editable install, trying the `test`, `tests`, `testing`, and `dev` extras in order, then a plain `-e .`; a uv "does not have an extra" warning is not counted as success
3. Test and development requirement files, which are additive and non-fatal — failures are recorded and installation continues

A repository with no `pyproject.toml`/`setup.py`/`setup.cfg` skips editable installation. Requirement files are discovered as the conventional `requirements-test.txt`, `requirements-dev.txt`, `test-requirements.txt`, `requirements/test*.txt`, and `requirements/tests.txt`, plus pattern matches for `requirements-test*`, `requirements-dev*`, `test-requirements*`, and nested `requirements/test*.txt`. The test command is rebound to the entry's interpreter: `python`/`python3` is replaced, and `pytest` becomes `python -m pytest`.

Closing an entry removes both the environment and the clone. A startup sweep also clears `.environments` plus the configured clone paths, so a crashed run cannot poison the next one; unrelated workspace directories are preserved.

### Baseline gate

Both `--validate` and normal runs classify a baseline test outcome before any model call. Only `fails_as_expected` releases the agent; every other status is recorded with `failure_class: infrastructure` and a human-readable reason.

### Resolution loop

The LangGraph loop runs at most three attempts. Each failed attempt is reverted, diagnosed, and summarized back into the next prompt with the previous reasoning, the tail of stdout and stderr (40 lines / 4000 characters per stream), and any timeout wording. A `wrong_file` diagnosis adds the file to a do-not-retry list. Proposals rejected by guardrails are recorded as guard events and retried without touching the repository; three rejections end the entry.

### Run statuses

| Status | Failure class | Meaning |
| --- | --- | --- |
| `success` | — | A fix passed the configured tests |
| `failed` | agent | The attempt budget was exhausted |
| `test_timeout` | agent | An agent test run timed out |
| `context_too_large` | agent | A request exceeded the configured input limit |
| `could_not_install` | infrastructure | The entry's environment could not be installed |
| `provider_unavailable` | infrastructure | The provider rejected a required feature or stayed unavailable |
| `errored` | infrastructure | Clone, baseline, checkout, or any other setup failure |

## Run reports

Every entry writes a JSON report to `runs/` (`<timestamp>_<owner>-<repository>_<issue>.json`, schema version 4) recording:

- repository, issue, commit, provider, model, and `base_url_host`
- status, failure class, status reason, and error text
- start/finish timestamps and duration
- attempt count, LLM call count, per-call latency and token usage, the largest preflight input count, the method that produced it, and the configured limit
- every LLM call with its purpose (`exploration`, `fix_generation`, `diagnosis`, `provider_preflight`)
- per-attempt records: target file, reasoning, a size-capped unified diff, test result with a capped output tail, diagnosis, retained avoid-list, and the exploration steps taken
- guard events, the applied test patch's file paths (never its contents), the baseline record, and environment metadata including install attempts and cleanup errors
- the PR URL when one was created

Prompts, full file contents, and API keys are never written to reports. When a provider omits usage metadata, token values are `null` — they are never estimated.

Each run also writes `<timestamp>_benchmark_summary.json` with per-status counts and the paths of every report included in the run, including entries resumed from earlier reports.

`--validate --results-file` writes a separate JSON-lines stream with one record per completed entry: schema version, repository, status, failure class, return code, detail, the last 20 output lines, and the environment summary.

## Guardrails

- **Path confinement** — every file read, write, or listing resolves through `resolve_repository_path`, which rejects any path escaping the cloned repository's root (`swe_agent/agent/paths.py`)
- **Proposal validation** — a fix is rejected before application if the LLM's proposed `file_path` does not exist in the repository, preventing hallucinated or stale paths from crashing the run
- **Protected test files** — the agent cannot modify files under `tests/`/`test/`, `test_*.py`, `*_test.py`, `conftest.py`, files named by the entry's test command, or files touched by a benchmark test patch (recorded as guard-event kind `test_patch_file`); reading them is allowed
- **Structured output** — fixes and diagnoses are returned as schema-validated JSON, not free text
- **Revert on failure** — a fix that fails its test command is restored to its original contents
- **Bounded execution** — test commands are capped, the resolution loop runs at most three attempts, and three guard rejections end the entry, so a single issue cannot run indefinitely
- **Isolation and secret hygiene** — target repositories run in per-entry environments that inherit no API keys, and tokens are redacted from provider errors
- **Least-scope commits** — PR creation validates that only the fixed file is modified before staging, committing, and pushing to a dedicated `swe-agent/fix-issue-<n>` branch on a fork, ignoring untracked test caches

## Project layout

```
main.py                    CLI entry point (validate and benchmark modes)
streamlit_app.py           Streamlit UI
benchmark/
  repos.json               Committed benchmark configuration
  final_ids.txt            Selected benchmark entries
  patches/                 Committed test patches referenced by the config
  tools/                   Candidate builder, issue resolver, finalizer
src/swe_agent/
  llm.py                   Provider clients, tool schemas, retries, token accounting
  token_count.py           DeepSeek V4.1 preflight input-token counting
  process.py               Subprocess boundary with process-group timeouts
  git_runner.py            Timed git execution with a minimal environment
  agent/
    loop.py                LangGraph propose → test → diagnose loop
    fixer.py               Exploration tools + fix proposal
    executor.py            Apply, test, and revert fixes
    diagnose.py            Failure classification
    guardrails.py          Path exists, test-file, and patch-file protection
    pr.py                  Fork, commit, push, and PR creation
    paths.py               Path-traversal guardrail
    structured.py          Structured response parsing
  ingestion/               Repository cloning and GitHub issue fetching
  benchmark/
    config.py              Config loading and validation
    environment.py         Per-entry isolated environments and install plans
    baseline.py            Baseline failure classification
    test_patch.py          Test patch application and local baseline commit
    validate.py            Zero-LLM reproduction checks and JSONL streaming
    runner.py              Entry lifecycle, resume, and outcome statuses
    reporting.py           Safe run reports and benchmark summaries
tests/                     pytest suite
runs/                      Per-entry JSON reports (gitignored)
workspace/                 Clones, environments, and shared caches (gitignored)
```

## Scope

The current implementation targets Python repository issues supplied in benchmark JSON, and the agent proposes full-file rewrites (it does not emit partial hunk patches). It is a CLI and Streamlit workflow — there is no server backend, no parallel entry execution, and no hidden-test evaluation: an entry's tests are visible to the agent and are protected from modification.
