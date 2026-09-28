#!/usr/bin/env python3
"""Validate that a pull request title is a supported Conventional Commit."""

import argparse
import re


TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)
TITLE_PATTERN = re.compile(
    rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9][a-z0-9._/-]*\))?!?: \S.*$"
)


def validate(title: str) -> str | None:
    """Return an error for an invalid title, otherwise None."""
    if "\n" in title or "\r" in title:
        return "the title must be a single line"
    if not TITLE_PATTERN.fullmatch(title):
        return (
            "expected '<type>(optional-scope)[!]: description' with one of: "
            + ", ".join(TYPES)
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", help="pull request title to validate")
    args = parser.parse_args()

    error = validate(args.title)
    if error is not None:
        print(f"FAIL: invalid pull request title: {error}")
        print(f"Title: {args.title!r}")
        print("Example: feat(server): support constrained JSON output")
        return 1

    print(f"PASS: valid pull request title: {args.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
