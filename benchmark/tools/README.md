# Candidate builder

`build_candidates.py` turns the pinned `nebius/SWE-bench-extra` revision into
the config format used under `benchmark/candidates/`. It is standalone: nothing
under `src/swe_agent` imports it, and it never runs the agent or installs target
projects.

## Usage

```sh
python -m pip install -r benchmark/tools/requirements.txt
export GITHUB_TOKEN=...   # recommended; GraphQL needs it
python benchmark/tools/build_candidates.py --dry-run
python benchmark/tools/build_candidates.py --limit 150
```

The default dry run screens the newest 300 shape-filtered rows. Output goes to
`benchmark/candidates/` (gitignored): `repos.json`, test patches under
`patches/`, and JSON/Markdown reports. Gold fixes are written under `gold/` for
local inspection only; `repos.json` never references them. Use `--force` to
replace a non-empty output directory.

## Offline cache and token

Dataset rows are cached under the user cache directory, or `--cache-dir`.
GitHub resolution is cached in `<cache-dir>/issue_cache.json`. With `--offline`,
both caches must already exist; a PR missing from the issue cache is counted as
unresolved and excluded. No network calls are made in offline mode.

The resolver prefers GraphQL `closingIssuesReferences`, then falls back to
closing keywords in the PR body via REST. It only keeps same-repo references,
requires the live issue title to match the dataset problem statement, and
rejects references that `/issues/{n}` reports as pull requests. Set
`GITHUB_TOKEN` to use GraphQL and avoid low unauthenticated rate limits.

## Heavy repositories

`heavy_repos.txt` lists repositories whose test environments need heavy or
system-level dependencies. Add one `owner/repo` per line; matching is
case-insensitive. Denylisted rows are counted in the report.

## Validation

The builder only performs metadata and shape screening. The real correctness
gate is the existing benchmark validator, which checks out each pinned commit,
applies the test patch, and confirms the tests reproduce the failure:

```sh
uv run main.py --config benchmark/candidates/repos.json --validate
```

The builder deliberately does not invoke `--validate` itself.

## Finalizing the committed set

`final_ids.txt` lists the selected `owner/repository#issue` entries. After the
candidate output has been validated, regenerate the committed config and test
patches from the repository root:

```sh
python benchmark/tools/finalize_benchmark.py
```

The script copies only the referenced test patches from the gitignored
`benchmark/candidates/` tree. It rejects missing or duplicate ids, unexpected
output files, and patches that touch anything other than test files. Gold
patches are never read, copied, or referenced.
