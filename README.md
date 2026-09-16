# Autonomous SWE Coding Agent

An autonomous Python software-engineering agent. Point it at a GitHub repository, pick an issue from a Streamlit dropdown, and it investigates only the code relevant to that issue (never the whole repo — this saves tokens and keeps context tight), generates and applies a fix, runs tests in a Docker sandbox, analyzes failures, retries when needed, and opens a GitHub PR once the fix is verified.

## Architecture

```
Repo URL → Fetch Open Issues → Select Issue (Streamlit dropdown)
        ↓
   LangGraph Agent
        ↓
Explore Codebase (targeted, not full-repo)
        ↓
Form Hypothesis → Generate/Apply Patch
        ↓
Security Policy Check
        ↓
Docker Sandbox → pytest
        ↓
Failure? → Analyze → Revise → Retry (budget-limited)
        ↓
Success → Full Test Suite
        ↓
Git Commit → GitHub PR
```

ReAct-style loop: **Reason → Act → Observe**.

## Key Components

- **LangGraph** — agent orchestration and state
- **Groq (`openai/gpt-oss-120b`)** — initial LLM, behind a provider abstraction for easy swapping
- **Repository tools** — `search_code`, `read_file`, `find_references`, `run_tests`, `get_git_diff`, `apply_patch`
- **Docker** — isolated code execution with resource limits and restricted network
- **pytest** — targeted tests first, full suite before PR
- **Security layer** — tool permissions, secret protection, path/execution controls, prompt-injection defense (detailed below)
- **SQLite** — run state, iterations, patches, test results, security events, audit log
- **FastAPI** — backend
- **Streamlit** — issue selection + live execution trace
- **GitHub API** — issue listing, branch, commit, push, PR creation

The entire repository is never dumped into LLM context — the agent selectively explores only what's relevant to the selected issue.

## Security & Guardrails

Everything originating from the target repository — source code, README, comments, existing tests, test *output*, and the issue text itself — is treated as **untrusted, potentially adversarial input**. A malicious repo or a maliciously-crafted issue is an expected threat model, not an edge case.

**Prompt injection / jailbreak defense**
- Issue text, file contents, and test output are never concatenated directly into a "trusted" instruction context — they are passed to the LLM as clearly-delimited *data*, not as instructions
- The agent's system prompt and tool-permission set cannot be overridden by anything read from the repo, an issue body, a code comment, or test/stdout output, even if that content explicitly tries to ("ignore previous instructions", fake system tags, etc.)
- All LLM output is treated as a *proposed* action, never auto-executed — every tool call passes through the policy layer below before it runs
- Adversarial-input handling is measured separately, via the security benchmark (below), not just assumed

**Policy layer (sits between the LLM and every consequential tool call)**
- Explicit allow-list of permitted tool calls and permitted file paths per run
- No arbitrary shell/command execution — only the defined tool functions (`apply_patch`, `run_tests`, etc.) are callable, never a raw shell
- Path traversal blocked — file access confined to the cloned repo's working directory
- Secret/API-key detection on any content the agent tries to read, log, or include in a patch/PR — flagged and redacted before it leaves the sandbox
- Git/GitHub operations restricted to least-privilege credentials scoped to: create branch, commit, push (non-protected branches only), open PR — no merge, no delete, no admin scopes, no access to other repos
- Main/protected branches can never be written to directly

**Sandbox isolation**
- Every run executes in a disposable Docker container: CPU/memory limits, hard execution timeout, isolated filesystem
- No host credentials, no Docker socket, no privileged mode, network restricted or fully disabled
- Containers are destroyed after each run — nothing persists between issues except what's explicitly written to SQLite

**Loop and resource control**
- Hard cap on retry iterations and total wall-clock time per issue — no infinite loops, no runaway cost
- Every denied tool call and every detected security violation is logged with full context for later audit

## Evaluation

Two separate tracks — functional correctness and security are never mixed into one score.

**Functional benchmark** — a curated, pinned set of ~15–20 real Python bugs (real repos, fixed commit, known failing test, so results are reproducible), measuring:
- Fix/pass rate
- Iterations to fix (avg, median)
- Time to fix
- Timeout rate
- Retry-exhaustion rate
- Regression rate

**Adversarial security benchmark** — repos/issues deliberately crafted to attack the agent, measuring failure rate independently across:
- Prompt injection
- Secret exfiltration attempts
- Path traversal attempts
- Arbitrary code execution attempts
- Sandbox escape attempts
- Network abuse attempts
- Unauthorized Git/GitHub operation attempts
- Resource exhaustion attempts

Dev/tuning bugs are kept separate from the final held-out evaluation set for both benchmarks.

## Scope

Python repositories and bug fixing only. No RAG, vector databases, embeddings, or document retrieval.

**Investigate → Reason → Modify → Execute → Observe → Recover → Verify → PR**