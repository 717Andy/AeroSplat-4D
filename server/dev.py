"""One-command dev start for macOS, Linux, and Windows.

    python -m server.dev

Installs nothing. It expects the AeroSplat package, the API extras in
``server/requirements.txt``, and Node on ``PATH``. The demo episode is
generated the first time.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys

from .config import repo_root
from .demo import build_demo


def main() -> None:
    root = repo_root()
    demo = root / "episodes" / "demo" / "dataset.json"
    if not demo.is_file():
        print("generating episodes/demo from the tiny fixture")
        build_demo(root / "episodes" / "demo")
    web = root / "web"
    if not (web / "node_modules").is_dir():
        npm = _npm()
        print("installing frontend packages")
        subprocess.check_call([npm, "install"], cwd=web)
    api = subprocess.Popen([sys.executable, "-m", "server"], cwd=root)
    web_proc = subprocess.Popen([_npm(), "run", "dev", "--", "--host", "127.0.0.1"], cwd=web)
    print("API http://127.0.0.1:8000   UI http://127.0.0.1:5173")

    def _stop(*_args: object) -> None:
        for proc in (web_proc, api):
            if proc.poll() is None:
                proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while True:
            if api.poll() is not None:
                print(f"API exited with {api.returncode}")
                break
            if web_proc.poll() is not None:
                print(f"UI exited with {web_proc.returncode}")
                break
            try:
                api.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
    finally:
        for proc in (web_proc, api):
            if proc.poll() is None:
                proc.terminate()


def _npm() -> str:
    name = "npm.cmd" if sys.platform == "win32" else "npm"
    found = shutil.which(name) or shutil.which("npm")
    if found is None:
        raise SystemExit("npm is not on PATH. Install Node 20 or newer.")
    return found


if __name__ == "__main__":
    main()
