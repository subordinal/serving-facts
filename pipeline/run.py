"""Run the pipeline: extract to staging, build tables with DuckDB, check them, write Parquet and a report.

python -m pipeline.run --vllm PATH --sglang PATH --profiler PATH --known-issues PATH
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import duckdb

from . import extract

ROOT = Path(__file__).resolve().parent.parent
STAGING, TABLES = ROOT / "staging", ROOT / "tables"

# Table name -> the SELECT that builds it from staging. Types are fixed here, not inferred downstream.
BUILD = {
    "runtime_releases": """
        SELECT runtime, version, release_order::INT AS release_order, released::DATE AS released
        FROM stg_releases ORDER BY runtime, release_order""",
    "architecture_support": """
        SELECT a.runtime, a.version, r.release_order, a.architecture, a.support
        FROM stg_architectures a JOIN runtime_releases r USING (runtime, version)
        ORDER BY a.runtime, r.release_order, a.architecture""",
    "architecture_ranges": """
        SELECT runtime, architecture,
               arg_min(version, release_order) AS first_version, arg_max(version, release_order) AS last_version,
               count(*)::INT AS releases,
               arg_max(version, release_order) = (SELECT arg_max(version, release_order) FROM architecture_support s2 WHERE s2.runtime = s.runtime) AS in_latest
        FROM architecture_support s WHERE support = 'native' GROUP BY runtime, architecture ORDER BY runtime, architecture""",
    "known_issues": "SELECT * FROM stg_known_issues ORDER BY runtime, issue_id",
    "startup_runs": """
        SELECT run_id, source, source_path, runtime, version, node, experiment, iteration::INT AS iteration,
               model, varied_param, varied_value, engine_ready, total_s::DOUBLE AS total_s,
               process_start_s::DOUBLE AS process_start_s, engine_start_s::DOUBLE AS engine_start_s,
               model_load_s::DOUBLE AS model_load_s, engine_init_s::DOUBLE AS engine_init_s,
               weight_load_s::DOUBLE AS weight_load_s, compile_s::DOUBLE AS compile_s,
               compile_cache_hit_s::DOUBLE AS compile_cache_hit_s, graph_capture_s::DOUBLE AS graph_capture_s,
               model_memory_gib::DOUBLE AS model_memory_gib
        FROM stg_startup_runs ORDER BY run_id""",
}

# Each check is a query that returns the number of violating rows. Any non-zero count fails the run.
CHECKS = {
    "runtime_releases: one row per runtime and version": "SELECT count(*) FROM (SELECT 1 FROM runtime_releases GROUP BY runtime, version HAVING count(*) > 1)",
    "runtime_releases: release_order is dense per runtime": "SELECT count(*) FROM (SELECT runtime FROM runtime_releases GROUP BY runtime HAVING max(release_order) + 1 <> count(*))",
    "architecture_support: unique key": "SELECT count(*) FROM (SELECT 1 FROM architecture_support GROUP BY runtime, version, architecture HAVING count(*) > 1)",
    "architecture_support: support is native or transformers": "SELECT count(*) FROM architecture_support WHERE support NOT IN ('native', 'transformers')",
    "architecture_support: architecture names look like class names": "SELECT count(*) FROM architecture_support WHERE NOT regexp_matches(architecture, '^[A-Za-z_][A-Za-z0-9_]*$')",
    "architecture_support: every release has at least 5 architectures": "SELECT count(*) FROM (SELECT 1 FROM architecture_support GROUP BY runtime, version HAVING count(*) < 5)",
    "known_issues: unique id": "SELECT count(*) FROM (SELECT 1 FROM known_issues GROUP BY issue_id HAVING count(*) > 1)",
    "known_issues: url points at GitHub": "SELECT count(*) FROM known_issues WHERE url NOT LIKE 'https://github.com/%'",
    "known_issues: cause and symptom present": "SELECT count(*) FROM known_issues WHERE cause IS NULL OR symptom IS NULL",
    "startup_runs: unique run_id": "SELECT count(*) FROM (SELECT 1 FROM startup_runs GROUP BY run_id HAVING count(*) > 1)",
    "startup_runs: durations are not negative": "SELECT count(*) FROM startup_runs WHERE least(total_s, coalesce(process_start_s, 0), coalesce(engine_start_s, 0), coalesce(model_load_s, 0), coalesce(engine_init_s, 0)) < 0",
    "startup_runs: phases do not exceed the total": "SELECT count(*) FROM startup_runs WHERE engine_ready AND coalesce(process_start_s, 0) + coalesce(engine_start_s, 0) + coalesce(model_load_s, 0) + coalesce(engine_init_s, 0) > total_s + 1",
    "startup_runs: memory is plausible": "SELECT count(*) FROM startup_runs WHERE model_memory_gib IS NOT NULL AND (model_memory_gib <= 0 OR model_memory_gib > 2000)",
}


def stage(name, rows):
    path = STAGING / f"{name}.jsonl"
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"  staged {n:>7,} rows  {name}", file=sys.stderr)
    return path


def chain(*iterables):
    for it in iterables:
        yield from it


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True, help="path to a vLLM clone with tags")
    ap.add_argument("--sglang", required=True, help="path to an SGLang clone with tags")
    ap.add_argument("--profiler", required=True, help="path to a clone of upb-cn/vllm-startup-profiler")
    ap.add_argument("--known-issues", required=True, help="path to readyml's known_issues.json")
    args = ap.parse_args(argv)
    STAGING.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    print("extract", file=sys.stderr)
    staged = {
        "releases": stage("releases", chain(extract.releases("vllm", args.vllm), extract.releases("sglang", args.sglang))),
        "architectures": stage("architectures", chain(extract.vllm_architectures(args.vllm), extract.sglang_architectures(args.sglang))),
        "known_issues": stage("known_issues", extract.known_issues(args.known_issues)),
        "startup_runs": stage("startup_runs", extract.startup_runs(args.profiler)),
    }

    print("build", file=sys.stderr)
    db = duckdb.connect()
    for name, path in staged.items():
        db.execute(f"CREATE VIEW stg_{name} AS SELECT * FROM read_json_auto('{path}', format='newline_delimited', sample_size=-1)")
    counts = {}
    for table, sql in BUILD.items():
        db.execute(f"CREATE TABLE {table} AS {sql}")
        counts[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  built  {counts[table]:>7,} rows  {table}", file=sys.stderr)

    print("check", file=sys.stderr)
    results = {name: db.execute(sql).fetchone()[0] for name, sql in CHECKS.items()}
    failed = {n: v for n, v in results.items() if v}
    for name, violations in results.items():
        print(f"  {'FAIL' if violations else 'ok  '}  {name}" + (f"  ({violations} rows)" if violations else ""), file=sys.stderr)
    if failed:
        sys.exit(f"{len(failed)} checks failed; tables not written")

    for table in BUILD:
        db.execute(f"COPY {table} TO '{TABLES / (table + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
    report(db, counts, results)
    print(f"wrote {len(BUILD)} tables to {TABLES}", file=sys.stderr)


def report(db, counts, results):
    def q(sql):
        return db.execute(sql).fetchall()

    lines = [f"# Data report\n\nBuilt {date.today().isoformat()}. Generated by `pipeline/run.py`; do not edit by hand.\n", "## Tables\n", "| Table | Rows |", "|---|---|"]
    lines += [f"| `{t}` | {n:,} |" for t, n in counts.items()]
    lines += ["\n## Coverage\n", "| Runtime | Releases | First | Last | With architectures |", "|---|---|---|---|---|"]
    for row in q("""SELECT r.runtime, count(*), arg_min(version, release_order), arg_max(version, release_order),
                           (SELECT count(DISTINCT version) FROM architecture_support a WHERE a.runtime = r.runtime)
                    FROM runtime_releases r GROUP BY r.runtime ORDER BY 1"""):
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += ["\n## Startup runs\n", "| vLLM version | Runs | Engine ready | Median total (s) | Median engine init share |", "|---|---|---|---|---|"]
    for row in q("""SELECT coalesce(version, 'unknown'), count(*), count(*) FILTER (WHERE engine_ready),
                           round(median(total_s) FILTER (WHERE engine_ready), 1),
                           round(100 * median(engine_init_s / total_s) FILTER (WHERE engine_ready AND total_s > 0), 0) || '%'
                    FROM startup_runs GROUP BY 1 ORDER BY 2 DESC"""):
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += [f"\n## Checks\n\n{len(results)} checks, all passing.\n"] + [f"- {name}" for name in results]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
