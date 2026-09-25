#!/usr/bin/env python3
"""Verify SHA256SUMS of this extracted delivery; no network or writes."""
from __future__ import annotations
import hashlib
from pathlib import Path, PurePosixPath
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sums = root / 'SHA256SUMS'
    if not sums.is_file():
        raise SystemExit('SHA256SUMS missing; use the complete delivery ZIP.')
    checked = 0
    for line in sums.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        rel = PurePosixPath(relative)
        if rel.is_absolute() or '..' in rel.parts or '\\' in relative:
            raise SystemExit(f'Unsafe checksum path: {relative}')
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise SystemExit(f'Missing/escaped: {relative}')
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f'Checksum mismatch: {relative}')
        checked += 1
    for p in (root / 'baseline').glob('*.zip'):
        with zipfile.ZipFile(p) as archive:
            if archive.testzip() is not None:
                raise SystemExit(f'Bad nested ZIP CRC: {p.name}')
            for name in archive.namelist():
                q = PurePosixPath(name)
                if q.is_absolute() or '..' in q.parts or '\\' in name:
                    raise SystemExit(f'Unsafe nested ZIP entry: {name}')
    print(f'BUNDLE_INTEGRITY=PASS ({checked} SHA256 entries; baseline ZIP CRC checked)')


if __name__ == '__main__':
    main()
