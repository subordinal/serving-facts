import subprocess
import tempfile
import unittest
from pathlib import Path

from pipeline import extract

LOG = """INFO 07-17 14:20:20 [x.py:1] boot
INFO 07-17 14:20:30 [core.py:69] Initializing a V1 LLM engine (v0.9.2) with config: model='/models/m'
INFO 07-17 14:20:31 [gpu_model_runner.py:1775] Loading model from scratch...
INFO 07-17 14:20:46 [default_loader.py:272] Loading weights took 14.46 seconds
INFO 07-17 14:20:46 [gpu_model_runner.py:1801] Model loading took 24.2841 GiB and 14.707185 seconds
INFO 07-17 14:21:07 [backends.py:155] Directly load the compiled graph(s) for shape None from the cache, took 9.353 s
INFO 07-17 14:21:35 [gpu_model_runner.py:2326] Graph capturing finished in 23.18 secs, took 0.60 GiB
INFO 07-17 14:21:35 [core.py:172] init engine (profile, create kv cache, warmup model) took 49.05 seconds
"""


def repo_with_tags(root, files_by_tag):
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.test")
    run("config", "user.name", "t")
    for tag, files in files_by_tag.items():
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        run("add", "-A")
        run("commit", "-q", "--allow-empty", "-m", tag)
        run("tag", tag)


class ExtractTest(unittest.TestCase):
    def test_startup_run_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "experiments/A100/examples/max_model_len/iterations/3/output_model_llama2-13b-hf_max-model-len_256.txt"
            path.parent.mkdir(parents=True)
            path.write_text(LOG)
            rows = list(extract.startup_runs(d))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["node"], r["experiment"], r["iteration"]), ("A100", "max_model_len", 3))
        self.assertEqual((r["model"], r["varied_param"], r["varied_value"]), ("llama2-13b-hf", "max-model-len", "256"))
        self.assertEqual((r["version"], r["engine_ready"], r["total_s"]), ("0.9.2", True, 75.0))
        self.assertEqual((r["model_load_s"], r["engine_init_s"]), (15.0, 49.0))
        self.assertEqual((r["weight_load_s"], r["compile_cache_hit_s"], r["model_memory_gib"]), (14.46, 9.35, 24.2841))

    def test_sglang_architectures_per_tag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = "python/sglang/srt/models/"
            repo_with_tags(root, {
                "v0.1.0": {m + "llama.py": "class LlamaForCausalLM: pass\nEntryClass = LlamaForCausalLM\n"},
                "v0.2.0": {m + "qwen.py": "class A: pass\nclass TransformersForCausalLM: pass\nEntryClass = [A, TransformersForCausalLM]\n"},
                "v0.2.0rc1": {},
            })
            self.assertEqual(extract.release_tags(root), ["0.1.0", "0.2.0"])
            rows = {(r["version"], r["architecture"], r["support"]) for r in extract.sglang_architectures(root)}
        self.assertEqual(rows, {
            ("0.1.0", "LlamaForCausalLM", "native"), ("0.2.0", "LlamaForCausalLM", "native"),
            ("0.2.0", "A", "native"), ("0.2.0", "TransformersForCausalLM", "transformers"),
        })

    def test_vllm_architectures_skip_unreadable_releases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reg = "vllm/model_executor/models/registry.py"
            good = '_T = {"LlamaForCausalLM": ("llama", "LlamaForCausalLM")}\n_TRANSFORMERS_X = {"GemmaForCausalLM": ("t", "T")}\n_VLLM_MODELS = {**_T, **_TRANSFORMERS_X}\n'
            repo_with_tags(root, {"v0.1.0": {"README": "x"}, "v0.2.0": {reg: "X = 1\n"}, "v0.3.0": {reg: good}})
            rows = [(r["version"], r["architecture"], r["support"]) for r in extract.vllm_architectures(root)]
        self.assertEqual(rows, [("0.3.0", "LlamaForCausalLM", "native"), ("0.3.0", "GemmaForCausalLM", "transformers")])


if __name__ == "__main__":
    unittest.main()
