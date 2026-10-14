"""Bridge Interactive RSI between this laptop and a training run on a remote instance.

The editor (scripts/interactive_rsi.py --inject-rundir <LOCAL_MIRROR>) and the trainer talk through
small files in a run directory (trainer.py _check_pose_injection). This script keeps a local mirror
directory in sync with the remote run directory over ssh, only for those files:

  local -> remote : pose_injection_request.flag  (P / I pressed)   injection_edited_state.npz (I pressed)
  remote -> local : injection_current_state.npz (robot 0 when paused), live_view.npz (watch_training.py)

Usage (leave running in its own terminal):
    python scripts/rsi_bridge.py                      # defaults below
    python scripts/rsi_bridge.py --port 50908 --host root@108.240.82.27 \
        --remote-rundir /workspace/runs_dots64/r512/servo_gait_v3

Then, in other terminals:
    python scripts/interactive_rsi.py --inject-rundir ~/r2dreamer/runs/remote512/servo_gait_v3
    python scripts/watch_training.py  --rundir        ~/r2dreamer/runs/remote512/servo_gait_v3
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

FLAG, EDITED, CURRENT, LIVE = ("pose_injection_request.flag", "injection_edited_state.npz",
                               "injection_current_state.npz", "live_view.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="root@108.240.82.27")
    p.add_argument("--port", default="50908")
    p.add_argument("--remote-rundir", default="/workspace/runs_dots64/r512/servo_gait_v3")
    p.add_argument("--local-rundir", default=str(Path.home() / "r2dreamer/runs/remote512/servo_gait_v3"))
    args = p.parse_args()
    local = Path(args.local_rundir).expanduser()
    local.mkdir(parents=True, exist_ok=True)
    remote = args.remote_rundir
    ctl = str(Path.home() / ".ssh" / f"rsi_bridge_{args.port}.sock")
    ssh_opts = ["-o", "ControlMaster=auto", "-o", f"ControlPath={ctl}", "-o", "ControlPersist=15m",
                "-o", "ConnectTimeout=60", "-o", "IPQoS=none", "-o", "ServerAliveInterval=15",
                "-o", "StrictHostKeyChecking=accept-new", "-p", args.port]
    scp_opts = [o if o != "-p" else "-P" for o in ssh_opts]

    def ssh(cmd, timeout=90):
        r = subprocess.run(["ssh", *ssh_opts, args.host, cmd], capture_output=True, text=True, timeout=timeout)
        return r.returncode, "\n".join(l for l in r.stdout.splitlines()
                                       if not l.startswith(("Welcome", "Have fun", "AI agents")))

    def scp_to(src, dst):
        subprocess.run(["scp", *scp_opts, str(src), f"{args.host}:{dst}"], check=True, capture_output=True, timeout=90)

    def scp_from(src, dst):
        subprocess.run(["scp", *scp_opts, f"{args.host}:{src}", str(dst)], check=True, capture_output=True, timeout=90)

    print(f"RSI bridge: {local}  <->  {args.host}:{args.port}:{remote}", flush=True)
    rc, out = ssh(f"test -d {remote} && echo OK")
    if "OK" not in out:
        sys.exit(f"cannot reach the remote run directory {remote} (ssh said: {out!r})")
    pending, flag_time, last_live, last_state_mtime = False, 0, 0., 0
    while True:
        try:
            now = time.time()
            if (local / EDITED).exists():
                scp_to(local / EDITED, f"{remote}/{EDITED}.tmp")
                ssh(f"mv {remote}/{EDITED}.tmp {remote}/{EDITED}")
                (local / EDITED).unlink(missing_ok=True)
                print(time.strftime("%H:%M:%S"), "pose sent to the run", flush=True)
            if (local / FLAG).exists() and not pending:
                rc, out = ssh(f"touch {remote}/{FLAG}; date +%s")
                flag_time, pending = int(out.split()[-1]) if out.split() else int(now), True
                print(time.strftime("%H:%M:%S"), "pause requested on the run", flush=True)
            rc, out = ssh(f"test -e {remote}/{FLAG} && echo FLAG; stat -c %Y {remote}/{CURRENT} 2>/dev/null")
            remote_flag = "FLAG" in out
            nums = [t for t in out.split() if t.isdigit()]
            state_mtime = int(nums[0]) if nums else 0
            if pending and state_mtime >= flag_time and state_mtime != last_state_mtime and not (local / CURRENT).exists():
                scp_from(f"{remote}/{CURRENT}", local / CURRENT)
                last_state_mtime = state_mtime
                print(time.strftime("%H:%M:%S"), "robot 0 loaded: it is paused, pose it, then press I", flush=True)
            if pending and not remote_flag and not (local / EDITED).exists():
                (local / FLAG).unlink(missing_ok=True)
                pending = False
                print(time.strftime("%H:%M:%S"), "injection applied; training resumed", flush=True)
            if now - last_live > 3.:
                try:
                    scp_from(f"{remote}/{LIVE}", local / (LIVE + ".tmp"))
                    (local / (LIVE + ".tmp")).replace(local / LIVE)
                except Exception:
                    pass
                last_live = now
        except subprocess.TimeoutExpired:
            print("ssh timed out; retrying", flush=True)
        except Exception as exc:  # keep the bridge alive through transient network errors
            print(f"bridge error ({type(exc).__name__}): {exc}", flush=True)
            time.sleep(3)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
