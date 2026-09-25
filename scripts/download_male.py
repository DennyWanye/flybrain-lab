#!/usr/bin/env python3
"""Download fly.ai brain-v1 assets, checking even already-existing files.
Hashes pinned from fly.ai 95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e/flybrain/data.py.
Data are not included in this starter kit. Requires network access on your machine.
"""
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

URL = "https://github.com/alextitonis/fly.ai/releases/download/brain-v1"
FILES = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/male")
    a = p.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        if target.exists():
            if digest(target) == expected:
                print(f"Verified existing: {target}")
                continue
            raise RuntimeError(f"Checksum mismatch in {target}; quarantine it, then retry. Not overwriting.")
        temporary = root / (name + ".part")
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(f"{URL}/{name}", headers={"User-Agent": "FlyBrainLab/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as f:
                    while block := response.read(1 << 20):
                        f.write(block)
                if digest(temporary) != expected:
                    raise RuntimeError(f"Downloaded {name} has the wrong SHA256. Not using it.")
                temporary.replace(target)
                print(f"Downloaded and verified: {target}")
                break
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                print(f"Attempt {attempt} failed: {exc}; retrying")
                time.sleep(attempt * 3)
    (root / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in FILES.items()))


if __name__ == "__main__":
    main()
