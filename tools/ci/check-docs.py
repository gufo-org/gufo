#!/usr/bin/env python3
"""
check-docs.py — Documentation integrity, local link, and syntax validator for gufo

Validates:
1. Presence of maintained user/developer documentation and license notices.
2. Local Markdown links and section anchor integrity without network access.
3. Syntax validity of fenced JSON code blocks.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REQUIRED_DOC_FILES = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "AGENTS.md",
    "docs/CLI.md",
    "docs/DEVELOPMENT.md",
    "docs/models/README.md",
    "docs/SERVER.md",
    "docs/TESTING.md",
    "docs/BENCHMARKS.md",
    "docs/PERFORMANCE.md",
    *[f"docs/models/{model}/{document}.md"
      for model in ("deepseek-v4-flash", "qwen3.8-27b", "qwen3.8-flash-next",
                    "qwen3-asr", "qwen3-tts", "qwen-image-2.1", "minimax-h3")
      for document in ("README", "BENCHMARKS", "QUALITY", "EXPERIMENTS")],
]

EXCLUDE_DIRS = {
    ".git", ".jj", ".direnv", "build", "Build", "result", "node_modules", ".cache",
    "llama.cpp", "ds4"
}

# Regex to extract Markdown links: [label](target)
MD_LINK_REGEX = re.compile(r'\[(?P<label>[^\]]+)\]\((?P<target>[^\)\s]+)\)')

# Regex to extract markdown headings: # Heading
MD_HEADING_REGEX = re.compile(r'^(?P<level>#{1,6})\s+(?P<title>.+)$', re.MULTILINE)

# Regex to extract fenced code blocks: ```lang ... ```
FENCED_BLOCK_REGEX = re.compile(r'```(?P<lang>\w+)?\n(?P<code>.*?)```', re.DOTALL)


def slugify_heading(title: str) -> str:
    """Convert heading title to GitHub markdown anchor slug."""
    # Remove inline markdown links/formatting [foo](bar) -> foo, `code` -> code
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
    t = re.sub(r'[`*_]', '', t)
    # Strip HTML tags
    t = re.sub(r'<[^>]+>', '', t)
    # Lowercase
    slug = t.strip().lower()
    # Replace non-alphanumeric (except dashes and spaces)
    slug = re.sub(r'[^\w\s-]', '', slug)
    # Replace spaces and consecutive dashes with a single dash
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug


def extract_anchors(content: str) -> Set[str]:
    """Extract all heading anchor slugs from markdown text."""
    anchors: Set[str] = set()
    for match in MD_HEADING_REGEX.finditer(content):
        title = match.group("title").strip()
        slug = slugify_heading(title)
        if slug:
            anchors.add(slug)
    return anchors


def find_markdown_files(repo_root: Path) -> List[Path]:
    """Find all tracked/relevant markdown files in the repository."""
    md_files: List[Path] = []
    for root, dirs, files in os_walk_filtered(repo_root):
        for f in files:
            if f.endswith(".md"):
                md_files.append(Path(root) / f)
    return md_files


def os_walk_filtered(root_path: Path):
    import os
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("build-")]
        yield root, dirs, files


def check_required_files(repo_root: Path) -> List[str]:
    """Check that all required documentation files exist."""
    errors: List[str] = []
    for req in REQUIRED_DOC_FILES:
        target = repo_root / req
        if not target.exists():
            errors.append(f"Missing required documentation file: {req}")
    return errors


def check_json_blocks(md_file: Path, content: str) -> List[str]:
    """Validate JSON syntax in fenced ```json code blocks."""
    errors: List[str] = []
    for match in FENCED_BLOCK_REGEX.finditer(content):
        lang = (match.group("lang") or "").lower()
        if lang == "json":
            code = match.group("code").strip()
            # If the block contains placeholder ellipsis comments, skip strict JSON parse
            if "..." in code or "/*" in code or "//" in code:
                continue
            try:
                json.loads(code)
            except Exception as e:
                # Find approximate line number
                line_no = content[: match.start()].count("\n") + 1
                errors.append(f"{md_file}:{line_no}: Invalid JSON in fenced code block: {e}")
    return errors


def check_links_and_anchors(repo_root: Path, md_files: List[Path]) -> Tuple[int, List[str]]:
    """Validate local links and anchor targets across markdown files."""
    errors: List[str] = []
    total_links_checked = 0

    # Cache extracted anchors per file path
    anchors_cache: Dict[Path, Set[str]] = {}

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{md_file}: Failed to read markdown file: {e}")
            continue

        # Check JSON blocks
        errors.extend(check_json_blocks(md_file, content))

        # Check links
        lines = content.splitlines()
        for line_idx, line in enumerate(lines, start=1):
            for match in MD_LINK_REGEX.finditer(line):
                target = match.group("target").strip()
                # Ignore external, web, mail, file:// links
                if (
                    target.startswith("http://")
                    or target.startswith("https://")
                    or target.startswith("mailto:")
                    or target.startswith("file://")
                    or target.startswith("conversation://")
                ):
                    continue

                total_links_checked += 1
                file_part = target
                anchor_part = ""

                if "#" in target:
                    file_part, anchor_part = target.split("#", 1)

                # Resolve file target
                if file_part:
                    # Resolve relative to current md file
                    resolved_target = (md_file.parent / file_part).resolve()
                    # Also try relative to repo root if not found
                    if not resolved_target.exists():
                        resolved_root = (repo_root / file_part).resolve()
                        if resolved_root.exists():
                            resolved_target = resolved_root
                    
                    if not resolved_target.exists():
                        errors.append(
                            f"{md_file}:{line_idx}: Broken local link to non-existent file '{file_part}'"
                        )
                        continue
                    target_file_for_anchor = resolved_target
                else:
                    # Same-file anchor link (#anchor)
                    target_file_for_anchor = md_file

                # If anchor is specified and target is a markdown file, verify anchor
                if anchor_part and target_file_for_anchor.suffix.lower() == ".md" and target_file_for_anchor.exists():
                    if target_file_for_anchor not in anchors_cache:
                        try:
                            t_content = target_file_for_anchor.read_text(encoding="utf-8")
                            anchors_cache[target_file_for_anchor] = extract_anchors(t_content)
                        except Exception:
                            anchors_cache[target_file_for_anchor] = set()

                    valid_anchors = anchors_cache[target_file_for_anchor]
                    norm_anchor = anchor_part.lower().strip()
                    if norm_anchor not in valid_anchors:
                        errors.append(
                            f"{md_file}:{line_idx}: Broken anchor '#{anchor_part}' in '{target_file_for_anchor.name}'"
                        )

    return total_links_checked, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate documentation integrity, links, anchors, and syntax."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root directory",
    )
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    all_errors: List[str] = []

    # 1. Required files
    all_errors.extend(check_required_files(repo_root))

    # 2. Find and check all markdown files
    md_files = find_markdown_files(repo_root)
    links_count, link_errors = check_links_and_anchors(repo_root, md_files)
    all_errors.extend(link_errors)

    if all_errors:
        print(f"\n[check-docs] FAILED: Found {len(all_errors)} documentation errors:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(
        f"[check-docs] PASSED: {len(md_files)} markdown files and {links_count} local links/anchors "
        f"verified with zero broken links or invalid JSON blocks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
