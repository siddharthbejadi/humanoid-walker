"""Inspect durable pipeline evidence without allocating Torch/JAX/GPU memory."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.script_free_run_ops import inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.run), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
