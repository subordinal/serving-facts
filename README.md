# serving-facts

Open tables of facts about model serving, built by a small pipeline you can rerun.

Running your own inference means answering questions nobody publishes as data. Which runtime release can load this model? Where does startup time go? What is already known to break? The answers live in source code, logs and issue threads. This repository turns them into tables.

```sql
-- Which releases load Qwen3 natively?
SELECT runtime, first_version, last_version, releases
FROM 'tables/architecture_ranges.parquet'
WHERE architecture = 'Qwen3ForCausalLM';
```

```
runtime  first_version  last_version  releases
sglang   0.4.6          0.5.23        28
vllm     0.8.4          0.34.0        44
```

## Tables

Parquet files in [`tables/`](tables). Row counts and coverage are in [REPORT.md](REPORT.md), which the pipeline regenerates on every run.

| Table | Rows | One row is | Source |
|---|---|---|---|
| `runtime_releases` | 162 | a tagged release of vLLM or SGLang | git tags |
| `architecture_support` | 20,631 | an architecture a release can load, native or through Transformers | the runtime's model registry, parsed at each tag |
| `architecture_ranges` | 757 | an architecture's first and last native release per runtime | derived |
| `known_issues` | 68 | a startup failure reported upstream | vLLM, SGLang and KServe issues, via [readyml](https://github.com/subordinal/readyml) |
| `startup_runs` | 2,028 | one measured vLLM startup, split into phases | logs published with an MLSys 2026 paper (see below) |

## What the data says

From [`queries/`](queries), run against the tables as published:

- **Half of startup is engine init.** Across 2,004 runs where the engine came up, the median split is 30% process start, 6% engine start, 12% model load and 50% engine init. Loading the weights is the small part.
- **A compile cache hit is about four times faster.** Median graph compilation is 16.6 s cold (181 runs) and 4.3 s on a cache hit (1,656 runs).
- **Runtimes drop models.** vLLM loaded `Phi3SmallForCausalLM` natively until 0.9.2 and `GemmaForCausalLM` until 0.33.0.

These describe this dataset, not inference in general. The startup runs come from one research group's hardware and mostly from vLLM 0.10.1.1.

## Query it

```
pip install duckdb
python -c "import duckdb; print(duckdb.sql(open('queries/where_startup_time_goes.sql').read()))"
```

Any engine that reads Parquet works.

## The pipeline

Three stages, in [`pipeline/`](pipeline):

1. **Extract** (`extract.py`): read each source and write flat rows to `staging/` as JSON lines. Parsing only.
2. **Build** (`run.py`): DuckDB SQL turns staging into typed tables. Every column type is set here.
3. **Check** (`run.py`): 13 queries, each counting rows that break a rule (unique keys, allowed values, durations that add up). Any violation stops the run before a table is written.

```
pip install duckdb git+https://github.com/subordinal/readyml
git clone --filter=blob:limit=300k --no-checkout https://github.com/vllm-project/vllm
git clone --filter=blob:limit=300k --no-checkout https://github.com/sgl-project/sglang
git clone --depth 1 https://github.com/upb-cn/vllm-startup-profiler
python -m pipeline.run --vllm vllm --sglang sglang --profiler vllm-startup-profiler \
    --known-issues path/to/readyml/src/readyml/data/known_issues.json
```

A full run takes about two minutes on a laptop. CI reruns the checks against the published tables on every push.

## Sources and credit

- **Startup runs** are parsed from the raw logs in [upb-cn/vllm-startup-profiler](https://github.com/upb-cn/vllm-startup-profiler) (Apache-2.0), published with "Breaking the Ice: Analyzing Cold Start Latency in vLLM" (MLSys 2026, [arXiv 2606.07362](https://arxiv.org/abs/2606.07362)). The measurements are theirs. The phase split is ours, read from log timestamps, and is not the paper's six-step breakdown.
- **Architecture support** is read from [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) source at each release tag.
- **Known issues** link to the upstream issue each one came from.

## Limits

- Architecture support says a release has an implementation registered. It does not say the model works on your GPU or with your flags.
- vLLM releases before 0.6.3 use a different registry layout and have no architecture rows.
- `startup_runs` are engine-only runs with no API server, so `total_s` ends when engine init finishes. Phase boundaries come from second-resolution log timestamps.
- Known issues were extracted from issue text. 16 of 68 are marked medium confidence, and only two have a confirmed fixed-in release.
- There is no table of launch flags per release yet. A first attempt searched the source for flag strings and reported removals that never happened, because vLLM generates most flags from config classes. It was dropped rather than published.

Apache-2.0. Built by [Subordinal](https://subordinal.github.io).
