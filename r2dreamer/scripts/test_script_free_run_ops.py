"""CPU process and archive tests, never training or physical gait evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import script_free_run_ops as ops


class RunOpsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.run = self.directory / "run"
        self.run.mkdir()
        self.journal = ops.RunJournal(self.run, "preflight", 0)

    def tearDown(self):
        self.temporary.cleanup()

    def test_atomic_json_is_fresh_unless_explicitly_replacing(self):
        path = self.directory / "record.json"
        ops.atomic_json(path, dict(value=1))
        with self.assertRaises(FileExistsError):
            ops.atomic_json(path, dict(value=2))
        ops.atomic_json(path, dict(value=3), replace=True)
        self.assertEqual(json.loads(path.read_text())["value"], 3)
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_actual_cpu_subprocess_has_durable_completed_stage_and_no_gpu_import(self):
        text = ops.run_stage([sys.executable, "-c", "print('CPU helper only')"], self.run / "worker.log", 10,
                             cwd=ROOT, env=dict(os.environ), journal=self.journal)
        self.assertIn("CPU helper only", text)
        record = ops.inspect_run(self.run)
        self.assertEqual(record["status"]["stage_status"], "completed")
        self.assertIsNone(record["status"]["child_pid"])
        self.assertFalse(record["checkpoint_contents_validated"])
        self.assertGreaterEqual(len((self.run / "stage_events.jsonl").read_text().splitlines()), 4)

    def test_actual_cpu_timeout_reaps_child_and_records_failure(self):
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            ops.run_stage([sys.executable, "-c", "import time; time.sleep(60)"], self.run / "timeout.log", .1,
                          cwd=ROOT, env=dict(os.environ), journal=self.journal, grace_seconds=.2)
        record = ops.inspect_run(self.run)["status"]
        self.assertEqual(record["stage_status"], "failed")
        self.assertIsNone(record["child_pid"])
        self.assertIsNotNone(record["returncode"])

    def test_posix_group_cleanup_targets_owned_group_then_waits(self):
        process = Mock(pid=123, returncode=-15)
        process.poll.return_value = -15
        with patch.object(ops.os, "name", "posix"), patch.object(ops.os, "killpg", create=True) as kill, \
             patch.object(ops.signal, "SIGKILL", 9, create=True):
            ops.terminate_owned_group(process, 1)
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args, (123, 9))
        self.assertEqual(process.wait.call_count, 2)

    def test_stop_exception_cleans_child_before_interrupted_journal(self):
        process = Mock(pid=123, returncode=-15)
        process.wait.side_effect = ops.PipelineStopped("Requested stop")
        with patch.object(ops.subprocess, "Popen", return_value=process), \
             patch.object(ops, "terminate_owned_group", return_value=False) as cleanup:
            with self.assertRaises(ops.PipelineStopped):
                ops.run_stage(["fixture"], self.run / "stop.log", 10,
                              cwd=ROOT, env={}, journal=self.journal)
        cleanup.assert_called_once_with(process, 120)
        self.assertEqual(ops.inspect_run(self.run)["status"]["stage_status"], "interrupted")

    def test_failed_returncode_or_overflow_never_passes_stage(self):
        for name, code in (("exit", "raise SystemExit(4)"), ("overflow", "print('contact overflow')")):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                ops.run_stage([sys.executable, "-c", code], self.run / (name + ".log"), 10,
                              cwd=ROOT, env=dict(os.environ), journal=self.journal)

    def test_backup_refuses_running_or_stale_crash_status(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            ops.backup_run(self.run, self.directory / "backup.tar.gz")

    def test_stopped_backup_roundtrip_hashes_every_payload_and_does_not_claim_resume(self):
        (self.run / "latest.pt").write_bytes(b"CPU fixture, not a learned checkpoint")
        self.journal.update(status="interrupted", child_pid=None)
        output = self.directory / "backup.tar.gz"
        report = ops.backup_run(self.run, output)
        self.assertEqual(report["sha256"], ops.sha256(output))
        self.assertFalse(report["exact_resume_supported"])
        with tarfile.open(output) as archive:
            manifest = json.load(archive.extractfile("BACKUP_MANIFEST.json"))
            self.assertEqual(manifest["files"]["latest.pt"], ops.sha256(self.run / "latest.pt"))
            self.assertEqual(archive.extractfile("run/latest.pt").read(), (self.run / "latest.pt").read_bytes())
        with self.assertRaises(ValueError):
            ops.backup_run(self.run, output)
        self.assertEqual(report["sha256"], ops.sha256(output))
        with self.assertRaises(ValueError):
            ops.backup_run(self.run, self.run / "nested.tar.gz")

    def test_changed_backup_input_removes_only_owned_partial_archive(self):
        self.journal.update(status="completed", child_pid=None)
        output = self.directory / "changed.tar.gz"
        real_hash = ops.sha256
        calls = {}
        def changed_hash(path):
            calls[path] = calls.get(path, 0) + 1
            return real_hash(path) if calls[path] == 1 else "changed"
        with patch.object(ops, "sha256", side_effect=changed_hash), self.assertRaises(RuntimeError):
            ops.backup_run(self.run, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
