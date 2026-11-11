#!/usr/bin/env python3
"""Verify locally acquired source files against the frozen manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(data_root: Path, manifest: Path):
    failures = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        matches = list(data_root.rglob(row["filename"]))
        if len(matches) != 1:
            failures.append((row["dataset"], row["filename"], f"found {len(matches)} files"))
            continue
        actual = digest(matches[0])
        if actual != row["sha256"]:
            failures.append((row["dataset"], row["filename"], f"sha256 {actual}"))
        else:
            print(f"OK\t{row['dataset']}\t{matches[0]}")
    if failures:
        for item in failures:
            print("FAIL\t" + "\t".join(item))
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("checksums.tsv"))
    args = parser.parse_args()
    verify(args.data_root, args.manifest)


if __name__ == "__main__":
    main()

