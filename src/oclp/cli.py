"""Command-line interface for the OCLP reference SDK."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from oclp.canonical import record_digest
from oclp.validation import load_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oclp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate OCLP JSON records")
    validate.add_argument("paths", nargs="+", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    failures = 0
    for path in args.paths:
        try:
            record = load_record(path)
        except (OSError, ValueError, ValidationError) as error:
            failures += 1
            print(f"{path}: invalid: {error}")
            continue
        print(f"{path}: valid {record.kind} {record.id} {record_digest(record)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
