"""CPU tests of capacity admission/error paths with mocked trainer processes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import teacher_capacity_probe as probe


def check_probe():
    checks = []
    checkpoint = dict(step=64000, optimizer_updates=32, finite_agent=True,
                      stop_reason="optimizer_update_limit")
    trials = [{"start_pose_id": pose} for _ in range(4) for pose in range(5)]
    evaluations = [dict(step=0, trials=trials), dict(step=64000, trials=trials)]
    probe.validate_case(checkpoint, evaluations, 32)
    checks.append("complete_matching_initial_and_final_evaluations_pass")

    for name, replacement in (
            ("missing_updates_rejected", {"optimizer_updates": 31}),
            ("signal_stop_rejected", {"stop_reason": "SIGTERM"}),
            ("nonfinite_weights_rejected", {"finite_agent": False}),
            ("missing_step_rejected", {"step": 0})):
        try:
            probe.validate_case(checkpoint | replacement, evaluations, 32)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    for name, records in (
            ("missing_initial_evaluation_rejected", evaluations[1:]),
            ("nonzero_initial_step_rejected", [dict(step=1, trials=trials), evaluations[-1]]),
            ("mismatched_final_step_rejected", [evaluations[0], dict(step=63999, trials=trials)]),
            ("missing_trials_rejected", [evaluations[0], dict(step=64000)]),
            ("malformed_trial_rejected", [evaluations[0], dict(step=64000, trials=[None] * 20)]),
            ("missing_pose_category_rejected", [evaluations[0], dict(step=64000, trials=[{"start_pose_id": 0}] * 20)])):
        try:
            probe.validate_case(checkpoint, records, 32)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    with tempfile.TemporaryDirectory(prefix="teacher_probe_control_checks_") as temporary:
        root = Path(temporary)
        blocked = root / "existing_file"
        blocked.write_text("preserve this file", encoding="utf-8")
        for name, args in (
                ("output_file_rejected", ["--output", str(blocked), "--dry-run"]),
                ("invalid_timeout_rejected", ["--output", str(root / "bad_timeout"),
                                              "--timeout-seconds", "0", "--dry-run"])):
            with patch.object(sys, "argv", ["teacher_capacity_probe.py", *args]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    probe.main()
                except SystemExit as exc:
                    if exc.code != 2:
                        raise
                    checks.append(name)
                else:
                    raise AssertionError(name)
        if blocked.read_text(encoding="utf-8") != "preserve this file":
            raise AssertionError("Output guard modified an existing file")

        dry_output = root / "dry_run"
        with patch.object(sys, "argv", ["teacher_capacity_probe.py", "--output",
                                       str(dry_output), "--dry-run"]), \
                patch.object(probe.subprocess, "Popen", side_effect=AssertionError("dry run launched trainer")), \
                redirect_stdout(io.StringIO()):
            probe.main()
        if dry_output.exists():
            raise AssertionError("Dry run wrote an output directory")
        checks.append("dry_run_launches_nothing_and_writes_nothing")

        for mode in ("pass", "timeout", "overflow"):
            output = root / mode
            calls = []

            class Process:
                def __init__(self, command, *, cwd, env, stdout, stderr):
                    self.command = command
                    self.waits = 0
                    calls.append("started")
                    run = Path(env["TEACHER_RUN_DIR"])
                    (run / "latest.pt").write_bytes(b"mock checkpoint, never a model")
                    (run / "walking_evaluations.jsonl").write_text(
                        "\n".join(json.dumps(record) for record in evaluations), encoding="utf-8")
                    stdout.write("LEARNING VERIFIED: MOCK TEST FIXTURE ONLY\n")
                    if mode == "overflow":
                        stdout.write("contact buffer overflow\n")

                def wait(self, timeout=None):
                    self.waits += 1
                    if mode == "timeout" and self.waits == 1:
                        raise subprocess.TimeoutExpired(self.command, timeout)
                    return 0

                def terminate(self):
                    calls.append("terminated")

                def kill(self):
                    calls.append("killed")

            args = ["teacher_capacity_probe.py", "--worlds", "64", "128",
                    "--updates", "32", "--output", str(output)]
            with patch.object(sys, "argv", args), patch.object(probe.subprocess, "Popen", Process), \
                    patch.object(probe, "read_checkpoint", return_value=copy.deepcopy(checkpoint)), \
                    redirect_stdout(io.StringIO()):
                try:
                    probe.main()
                except SystemExit as exc:
                    if mode == "pass" or exc.code != 1:
                        raise
            cases = json.loads((output / "capacity_report.json").read_text())["cases"]
            if mode == "pass":
                if len(cases) != 2 or any(case["status"] != "passed_short_capacity_only" for case in cases):
                    raise AssertionError("Mock valid capacity case did not pass")
                checks.append("valid_fixture_records_complete_short_capacity_pass")
            else:
                if len(cases) != 1 or cases[0]["status"] != "failed" or calls.count("started") != 1:
                    raise AssertionError("Failed probe did not stop sweep and retain failure report")
                if mode == "timeout" and "terminated" not in calls:
                    raise AssertionError("Timed out process was not terminated")
                checks.append(f"{mode}_retains_failure_and_stops_sweep")

    return dict(status="passed", check_count=len(checks), checks=checks,
                scope="CPU artifact validation and mocked subprocess control only; no GPU, learner or training executed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check_probe()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
