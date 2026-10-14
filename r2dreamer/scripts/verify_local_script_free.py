"""Record CPU software checks; this command never admits GPU training or walking."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--jax-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Select a fresh evidence directory")
    args.output.mkdir(parents=True)
    commands = [("all_cpu_tests", [str(args.torch_python), "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"], {}),
                ("jax_full_command", [str(args.jax_python), "scripts/test_full_command_env.py"], {"FULL_COMMAND_TEST_JAX": "1"}),
                ("jax_reference_resets", [str(args.jax_python), "scripts/test_reference_resets.py"], {"REFERENCE_RESET_TEST_JAX": "1"}),
                ("jax_clearance", [str(args.jax_python), "scripts/test_step_clearance.py"], {"STEP_CLEARANCE_TEST_JAX": "1"}),
                ("source_structure", [str(args.torch_python), "scripts/run_script_free_pipeline.py", "--mode", "structural"], {})]
    report = dict(status="failed", gpu_tested=False, learned_walking_tested=False,
                  scope="CPU contracts, real tiny synthetic optimizer tests and source structure; no physical/GPU qualification", cases=[])
    try:
        for name, command, extra in commands:
            started = time.monotonic()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", **extra)
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            output = result.stdout + "\n" + result.stderr
            (args.output / (name + ".log")).write_text(output, encoding="utf-8")
            count = re.search(r"Ran (\d+) tests? in", output)
            skip = re.search(r"skipped=(\d+)", output)
            row = dict(name=name, command=command, returncode=result.returncode,
                       seconds=time.monotonic() - started, log=name + ".log",
                       tests=None if count is None else int(count.group(1)),
                       skipped=0 if skip is None else int(skip.group(1)))
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            if result.returncode:
                raise RuntimeError(f"Local check {name} failed; inspect its preserved log")
        parsed = []
        for path in ROOT.rglob("*.py"):
            if any(part in (".venv", "__pycache__", "logdir") for part in path.relative_to(ROOT).parts):
                continue
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            parsed.append(path.relative_to(ROOT).as_posix())
        report.update(status="passed_local_software_only", parsed_python_files=parsed)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "local_verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
