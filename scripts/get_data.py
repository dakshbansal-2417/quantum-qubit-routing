"""Download the QASMBench circuits used in this project (small + medium suites).

    python scripts/get_data.py

Fetches exactly the QASMBench commit the results were produced with, and only the
`small` and `medium` folders (a few MB instead of the whole repository).
Needs `git` on the PATH. The hardware data (IBM FakeBrisbane) ships with the
qiskit-ibm-runtime package, so nothing else has to be downloaded.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/pnnl/QASMBench.git"
COMMIT = "357b942396d5c2b7cbc1c229c585a6ef5ccaebac"
DEST = Path(__file__).resolve().parents[1] / "data" / "QASMBench"


def git(*args, cwd=DEST):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main():
    if shutil.which("git") is None:
        sys.exit("git is not installed: get it from https://git-scm.com/downloads")
    if (DEST / "small").is_dir() and (DEST / "medium").is_dir():
        print(f"QASMBench already present in {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("remote", "add", "origin", REPO)
    git("sparse-checkout", "set", "small", "medium")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    git("checkout", "-q", "FETCH_HEAD")
    n = len(list(DEST.glob("small/*/*.qasm"))) + len(list(DEST.glob("medium/*/*.qasm")))
    print(f"Downloaded {n} circuit files to {DEST}")


if __name__ == "__main__":
    main()
