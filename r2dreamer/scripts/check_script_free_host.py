"""Record Linux setup prerequisites and advisory resources without importing GPU libraries."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
REQUIRED_TOOLS = ("python3", "curl", "nvidia-smi", "tar", "sha256sum")


def linux_memory():
    fields = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                fields[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return dict(total_bytes=fields.get("MemTotal"), available_bytes=fields.get("MemAvailable"))


def check_host(workspace, *, platform=None, python_version=None, lookup=shutil.which,
               runner=subprocess.run, memory_provider=linux_memory, disk_provider=shutil.disk_usage,
               writable_provider=lambda path: os.access(path, os.W_OK)):
    workspace = Path(workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("Setup workspace must be an existing directory")
    platform = sys.platform if platform is None else platform
    python_version = tuple(sys.version_info[:2]) if python_version is None else tuple(python_version[:2])
    if platform != "linux" or python_version < (3, 8):
        raise RuntimeError("Setup requires Linux and system Python 3.8+; the learner runtime will be Python 3.11")
    tools = {name: lookup(name) for name in REQUIRED_TOOLS}
    missing = [name for name, path in tools.items() if not path]
    if missing:
        raise RuntimeError("Missing setup prerequisites: " + ", ".join(missing))
    if not writable_provider(workspace):
        raise RuntimeError("Setup workspace is not writable")
    disk = disk_provider(workspace)
    if disk.free <= 0:
        raise RuntimeError("Setup workspace has no free disk space")
    result = runner([tools["nvidia-smi"], "--query-gpu=index,name,memory.total,driver_version",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != 4:
            raise RuntimeError("Malformed NVIDIA device inventory")
        index, name, memory_mib, driver = (item.strip() for item in row)
        devices.append(dict(index=int(index), name=name, total_vram_bytes=int(float(memory_mib) * 1024 ** 2),
                            driver_version=driver))
    if not devices or any(device["total_vram_bytes"] <= 0 for device in devices):
        raise RuntimeError("No usable NVIDIA device was reported by nvidia-smi")
    memory = memory_provider()
    warnings = []
    if max(device["total_vram_bytes"] for device in devices) < 32 * GIB:
        warnings.append("GPU VRAM is below the advisory 32 GiB plan; only full learner capacity can establish fit")
    if memory.get("total_bytes") is None or memory["total_bytes"] < 64 * GIB:
        warnings.append("Host RAM is unavailable or below the advisory 64 GiB plan")
    if disk.free < 100 * GIB:
        warnings.append("Free disk is below the advisory 100 GiB headroom plan; watch install/replay/checkpoint/video usage")
    if len(devices) != 1:
        warnings.append("Multiple NVIDIA devices are visible; use a declared single CUDA_VISIBLE_DEVICES selection")
    return dict(status="passed_linux_gpu_host_prerequisites_only",
                updated_at=datetime.now(timezone.utc).isoformat(), workspace=str(workspace),
                system_python=".".join(map(str, python_version)), tools=tools, gpu_inventory=devices,
                host_ram=memory, disk=dict(total_bytes=disk.total, free_bytes=disk.free),
                planning_estimates=dict(vram_bytes=32 * GIB, ram_bytes=64 * GIB, disk_headroom_bytes=100 * GIB,
                                        measured_minimums=False), warnings=warnings,
                gpu_runtime_tested=False, learner_capacity_tested=False, walking_qualified=False,
                scope="System prerequisites/device inventory only; pinned Torch/JAX, renderer and full learner admission still required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_host(args.workspace)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".host_report_", suffix=".tmp", dir=output.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
