"""Host setup admission tests use injected inventories, never a real GPU."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_script_free_host import check_host, GIB


class HostPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def check(self, **overrides):
        arguments = dict(platform="linux", python_version=(3, 8), lookup=lambda name: "/usr/bin/" + name,
            runner=lambda *args, **kwargs: SimpleNamespace(stdout="0, Test GPU, 49152, 580.0\n"),
            memory_provider=lambda: dict(total_bytes=64 * GIB, available_bytes=60 * GIB),
            disk_provider=lambda path: SimpleNamespace(total=500 * GIB, free=150 * GIB),
            writable_provider=lambda path: True)
        arguments.update(overrides)
        return check_host(self.root, **arguments)

    def test_inventory_records_resources_without_claiming_runtime_or_walking(self):
        report = self.check()
        self.assertEqual(report["status"], "passed_linux_gpu_host_prerequisites_only")
        self.assertEqual(report["gpu_inventory"][0]["total_vram_bytes"], 48 * GIB)
        self.assertEqual(report["warnings"], [])
        self.assertFalse(report["gpu_runtime_tested"])
        self.assertFalse(report["learner_capacity_tested"])
        self.assertFalse(report["walking_qualified"])
        self.assertFalse(report["planning_estimates"]["measured_minimums"])

    def test_missing_tools_or_readonly_workspace_fail_before_inventory(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Inventory must not run when setup cannot proceed")
        with self.assertRaisesRegex(RuntimeError, "curl"):
            self.check(lookup=lambda name: None if name == "curl" else name, runner=forbidden)
        with self.assertRaisesRegex(RuntimeError, "writable"):
            self.check(writable_provider=lambda path: False, runner=forbidden)

    def test_linux_and_system_python_are_required(self):
        for platform, version in (("win32", (3, 11)), ("linux", (3, 7))):
            with self.subTest(platform=platform, version=version), self.assertRaisesRegex(RuntimeError, "Linux"):
                self.check(platform=platform, python_version=version)

    def test_empty_or_malformed_gpu_inventory_is_rejected(self):
        for inventory in ("", "0, Unknown\n", "0, Empty GPU, 0, 580.0\n"):
            with self.subTest(inventory=inventory), self.assertRaises(RuntimeError):
                self.check(runner=lambda *args, **kwargs: SimpleNamespace(stdout=inventory))

    def test_resource_estimates_warn_but_do_not_fake_measured_capacity(self):
        report = self.check(runner=lambda *args, **kwargs: SimpleNamespace(stdout="0, Small GPU, 24576, 580.0\n"),
            memory_provider=lambda: dict(total_bytes=32 * GIB, available_bytes=20 * GIB),
            disk_provider=lambda path: SimpleNamespace(total=100 * GIB, free=40 * GIB))
        self.assertEqual(len(report["warnings"]), 3)
        self.assertFalse(report["learner_capacity_tested"])

    def test_no_disk_space_rejects_and_multiple_devices_require_declared_selection(self):
        with self.assertRaisesRegex(RuntimeError, "disk"):
            self.check(disk_provider=lambda path: SimpleNamespace(total=100 * GIB, free=0))
        report = self.check(runner=lambda *args, **kwargs: SimpleNamespace(
            stdout="0, GPU A, 49152, 580.0\n1, GPU B, 49152, 580.0\n"))
        self.assertTrue(any("Multiple" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
