"""Sequential job runner for Colab (or any remote box) that uploads results to a Hugging Face dataset repo.

Usage:
    python colab/runner.py --jobs colab/jobs-a100.json --hf-repo anthonym21/eve-rlcd-runs [--upload-weights]

Each job is {"name": ..., "cmd": "...", "eval": true|false}. `cmd` runs from the repo root with the
current interpreter substituted for a leading "python". After a job finishes, runs/<name>/ is uploaded
(logs, meta, policy.json, eval outputs; weights only with --upload-weights). Jobs whose results already
exist in the remote repo are skipped, so the runner can be restarted after a runtime reset.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

LOG_PATTERNS = ["*.json", "*.jsonl", "*.md", "*.png", "*.log", "*.txt"]
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "tokenizer*", "vocab*", "merges*", "special_tokens_map.json",
                   "adapter_*"]


def load_jobs(path: str) -> list[dict]:
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))
    for j in jobs:
        if "name" not in j or "cmd" not in j:
            raise ValueError(f"job needs name and cmd: {j}")
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate job names")
    return jobs


def run_cmd(cmd: str, log_path: Path) -> int:
    parts = shlex.split(cmd)
    if parts and parts[0] == "python":
        parts[0] = sys.executable
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(parts)}\n")
        log.flush()
        proc = subprocess.Popen(parts, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def _pid_alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def remote_done(api, repo: str, name: str) -> bool:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception:
        return False
    return f"runs/{name}/status.json" in files


def upload_run(api, repo: str, name: str, upload_weights: bool) -> None:
    folder = Path("runs") / name
    if not folder.exists():
        return
    patterns = LOG_PATTERNS + (WEIGHT_PATTERNS if upload_weights else [])
    api.upload_folder(folder_path=str(folder), path_in_repo=f"runs/{name}", repo_id=repo,
                      repo_type="dataset", allow_patterns=patterns)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--hf-repo", required=True, help="dataset repo id, created if missing (private)")
    ap.add_argument("--upload-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = ap.parse_args(argv)

    jobs = load_jobs(args.jobs)
    if args.dry_run:
        for j in jobs:
            print(j["name"], "::", j["cmd"])
        return 0

    lock = Path("runs") / "runner.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        old_pid = lock.read_text().strip()
        if _pid_alive(old_pid):
            print(f"[abort] another runner (pid {old_pid}) is still running; not starting a second one", flush=True)
            return 2
        print(f"[note] stale lock from pid {old_pid}; taking over", flush=True)
    lock.write_text(str(os.getpid()))

    from huggingface_hub import HfApi  # imported late so --dry-run needs no token
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)

    for job in jobs:
        name = job["name"]
        if remote_done(api, args.hf_repo, name):
            print(f"[skip] {name}: results already uploaded", flush=True)
            continue
        t0 = time.time()
        print(f"[start] {name}", flush=True)
        log_path = Path("runs") / name / "console.log"
        code = run_cmd(job["cmd"], log_path)
        status = {"name": name, "cmd": job["cmd"], "exit_code": code, "seconds": round(time.time() - t0, 1)}
        if code == 0 and job.get("eval"):
            for sub in ("run", "probe"):
                extra = f"python -m rlcd.eval {sub} --model runs/{name} --name {name} --out runs/{name}/eval"
                if sub == "run":
                    extra += " --split data/test.jsonl"
                status[f"eval_{sub}_exit"] = run_cmd(extra, log_path)
        (Path("runs") / name).mkdir(parents=True, exist_ok=True)
        if code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:] if log_path.exists() else []
            status["error_tail"] = tail
            print(f"[fail] {name} exit={code}; last log lines:", flush=True)
            for line in tail:
                print("    " + line, flush=True)
            print(f"[stop] {name} failed; later jobs may depend on it. Fix and rerun.", flush=True)
            lock.unlink(missing_ok=True)
            return code
        (Path("runs") / name / "status.json").write_text(json.dumps(status, indent=2))
        upload_run(api, args.hf_repo, name, args.upload_weights)
        print(f"[done] {name} exit={code} in {status['seconds']}s", flush=True)
    lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
