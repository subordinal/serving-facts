"""Extract stage: read each source and yield flat rows. No transformation beyond parsing."""
import ast
import json
import re
import subprocess
from pathlib import Path

from readyml import trace, vllm

RELEASE = re.compile(r"v(\d+\.\d+\.\d+)$")
SGLANG_MODELS = "python/sglang/srt/models"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace").stdout


def release_tags(repo):
    found = [m.group(1) for t in git(repo, "tag").split() if (m := RELEASE.match(t))]
    return sorted(found, key=lambda v: tuple(map(int, v.split("."))))


def releases(runtime, repo):
    for order, version in enumerate(release_tags(repo)):
        date = git(repo, "log", "-1", "--format=%cs", f"v{version}").strip()
        yield {"runtime": runtime, "version": version, "release_order": order, "released": date or None}


def vllm_architectures(repo):
    for version in release_tags(repo):
        source = git(repo, "show", f"v{version}:{vllm.REGISTRY}")
        if not source:
            continue
        try:
            registry = vllm.parse_registry(source)
        except (vllm.RuntimeSourceError, SyntaxError):
            continue
        for kind in ("native", "transformers"):
            for arch in sorted(registry[kind]):
                yield {"runtime": "vllm", "version": version, "architecture": arch, "support": kind}


def entry_classes(source):
    """SGLang registers every class named in a model file's module-level EntryClass."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "EntryClass":
            value = node.value
            elts = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            return [e.id for e in elts if isinstance(e, ast.Name)]
    return []


def sglang_architectures(repo):
    for version in release_tags(repo):
        files = [f for f in git(repo, "ls-tree", "-r", "--name-only", f"v{version}", SGLANG_MODELS).split("\n") if f.endswith(".py")]
        archs = set()
        for f in files:
            archs.update(entry_classes(git(repo, "show", f"v{version}:{f}")))
        for arch in sorted(archs):
            kind = "transformers" if arch.startswith("Transformers") else "native"
            yield {"runtime": "sglang", "version": version, "architecture": arch, "support": kind}


def known_issues(path):
    for r in json.loads(Path(path).read_text()):
        versions = r.get("versions") or {}
        yield {
            "issue_id": r["id"], "runtime": r["runtime"], "url": r["url"], "state": r["state"], "title": r["title"],
            "architectures": r.get("architectures") or [], "models": r.get("models") or [],
            "quantization": r.get("quantization") or [], "gpu_families": r.get("gpu_families") or [],
            "flags": r.get("flags") or [], "affected": versions.get("affected"), "fixed_in": versions.get("fixed_in"),
            "symptom": r.get("symptom"), "cause": r.get("cause"), "workaround": r.get("workaround"),
            "confidence": r.get("confidence"),
        }


RUN_FILE = re.compile(r"output_model_(?P<model>.+?)_(?P<param>[a-z0-9-]+)_(?P<value>[^_]+)\.txt$")
SOURCE = "upb-cn/vllm-startup-profiler"


def startup_runs(profiler_repo):
    """One row per startup log published with 'Breaking the Ice: Analyzing Cold Start Latency in vLLM' (MLSys 2026)."""
    root = Path(profiler_repo) / "experiments"
    for path in sorted(root.rglob("output_model_*.txt")):
        with open(path, errors="replace") as f:
            r = trace.parse(f)
        if r is None:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        name = RUN_FILE.match(path.name)
        iteration = parts[parts.index("iterations") + 1] if "iterations" in parts else None
        seconds = {p["phase"]: p["seconds"] for p in r["phases"]}
        rep = r["reported"]
        yield {
            "run_id": f"{SOURCE}:{rel.as_posix()}", "source": SOURCE, "source_path": rel.as_posix(),
            "runtime": "vllm", "version": r["version"], "node": parts[0],
            "experiment": parts[2] if len(parts) > 2 else None,
            "iteration": int(iteration) if iteration and iteration.isdigit() else None,
            "model": name.group("model") if name else None,
            "varied_param": name.group("param") if name else None,
            "varied_value": name.group("value") if name else None,
            "engine_ready": r["engine_ready"], "total_s": r["total_seconds"],
            "process_start_s": seconds.get("process start"), "engine_start_s": seconds.get("engine start"),
            "model_load_s": seconds.get("model load"), "engine_init_s": seconds.get("engine init"),
            "weight_load_s": rep.get("weight load"), "compile_s": rep.get("compile"),
            "compile_cache_hit_s": rep.get("compile cache hit"), "graph_capture_s": rep.get("graph capture"),
            "model_memory_gib": r["model_memory_gib"],
        }
