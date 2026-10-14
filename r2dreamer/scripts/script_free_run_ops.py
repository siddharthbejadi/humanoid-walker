"""Standard-library supervision and evidence-preserving run operations."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tarfile
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PipelineStopped(RuntimeError):
    pass


@contextmanager
def stop_signals():
    previous = {}

    def stop(signum, frame):
        raise PipelineStopped(f"Requested stop: {signal.Signals(signum).name}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class RunJournal:
    def __init__(self, directory, mode, seed):
        self.directory = Path(directory)
        self.record = dict(schema_version=1, status="running", mode=mode, seed=seed,
                           supervisor_pid=os.getpid(), host=socket.gethostname(),
                           started_at=utc_now(), walking_qualified=False,
                           replay_checkpointed=False, automatic_resume=False)
        self.update(stage="source_admission")

    def update(self, **fields):
        self.record.update(fields, updated_at=utc_now())
        atomic_json(self.directory / "run_status.json", self.record, replace=True)
        with (self.directory / "stage_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def terminate_owned_group(process, grace_seconds=120):
    """Allow a Linux learner to checkpoint, then reap the owned process group."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    forced = False
    if os.name == "nt" and process.poll() is None:
        # Windows CPU verification only; the rented Linux path has graceful TERM.
        forced = True
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=30, check=False)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        forced = True
    if os.name == "posix":
        # A parent may exit before its helpers. Kill any remaining owned group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    return forced


def run_stage(command, log, timeout, *, cwd, env, journal=None, grace_seconds=120):
    log = Path(log)
    fields = dict(stage=str(log.relative_to(journal.directory)) if journal else log.name,
                  command=list(command), log=str(log), timeout_seconds=timeout)
    if journal:
        journal.update(stage_status="starting", child_pid=None, **fields)
    options = {"start_new_session": True} if os.name == "posix" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    process = None
    try:
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, **options)
            if journal:
                journal.update(stage_status="running", child_pid=process.pid, **fields)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Stage exceeded {timeout}s; inspect {log}; no later stage admitted") from exc
        if code:
            raise RuntimeError(f"Stage exited {code}; inspect {log}; no later stage admitted")
        text = log.read_text(encoding="utf-8", errors="replace")
        if "overflow" in text.lower():
            raise RuntimeError(f"Overflow mentioned in {log}; contact/runtime evidence needs investigation")
    except BaseException as exc:
        # Repeated stop requests cannot interrupt cleanup and orphan the learner.
        previous = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, signal.SIG_IGN)
            forced = terminate_owned_group(process, grace_seconds) if process is not None else False
            if journal:
                journal.update(stage_status="interrupted" if isinstance(exc, (PipelineStopped, KeyboardInterrupt)) else "failed",
                               child_pid=None, returncode=None if process is None else process.returncode,
                               forced_termination=forced, error=f"{type(exc).__name__}: {exc}", **fields)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        raise
    else:
        terminate_owned_group(process, grace_seconds)
        if journal:
            journal.update(stage_status="completed", child_pid=None, returncode=code, error=None, **fields)
        return text


def inspect_run(directory):
    directory = Path(directory).resolve(strict=True)
    record = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    report_path = directory / "readiness_report.json"
    checkpoints = [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size}
                   for path in sorted(directory.rglob("*.pt")) if not path.is_symlink()]
    return dict(run=str(directory), status=record,
                readiness_report=json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None,
                checkpoint_files=checkpoints, checkpoint_contents_validated=False,
                note="A checkpoint file or completed updates do not establish walking. SIGKILL/crashes can leave stale running status.")


def backup_run(directory, output):
    directory = Path(directory).resolve(strict=True)
    output = Path(output).resolve()
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists() or output.is_relative_to(directory):
        raise ValueError("Use a fresh archive outside the run directory")
    inspection = inspect_run(directory)
    if inspection["status"].get("status") not in ("completed", "failed", "interrupted") or inspection["status"].get("child_pid") is not None:
        raise ValueError("Stop the pipeline and wait for a terminal run status before backup; stale crash status needs investigation")
    paths = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in run evidence: {path}")
        if path.is_file() and not path.name.endswith(".tmp"):
            paths.append(path)
    hashes = {path.relative_to(directory).as_posix(): sha256(path) for path in paths}
    manifest = dict(schema_version=1, created_at=utc_now(), source_run=str(directory), files=hashes,
                    replay_checkpointed=False, exact_resume_supported=False, walking_qualified=False)
    created = False
    sidecar_created = False
    try:
        with output.open("xb") as target:
            created = True
            with tarfile.open(fileobj=target, mode="w:gz") as archive:
                for path in paths:
                    archive.add(path, arcname="run/" + path.relative_to(directory).as_posix(), recursive=False)
                data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
                info = tarfile.TarInfo("BACKUP_MANIFEST.json")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        if any(sha256(path) != hashes[path.relative_to(directory).as_posix()] for path in paths):
            raise RuntimeError("Run evidence changed during backup; archive is not admitted")
        with tarfile.open(output, "r:gz") as archive:
            for name, expected in hashes.items():
                with archive.extractfile("run/" + name) as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                        raise RuntimeError(f"Backup readback mismatch: {name}")
        digest = sha256(output)
        with sidecar.open("x", encoding="ascii") as handle:
            sidecar_created = True
            handle.write(f"{digest}  {output.name}\n")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        if sidecar_created:
            sidecar.unlink(missing_ok=True)
        raise
    return dict(archive=str(output), sha256=digest, files=len(paths), bytes=output.stat().st_size,
                replay_checkpointed=False, exact_resume_supported=False, walking_qualified=False)
