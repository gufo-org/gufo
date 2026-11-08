#!/usr/bin/env python3
"""
check-docs.py — Documentation integrity, local link, and syntax validator for gufo

Validates:
1. Presence of maintained user/developer documentation and license notices.
2. Local Markdown links and section anchor integrity without network access.
3. Syntax validity of fenced JSON code blocks.
4. Public documentation rules from docs/AGENTS.md: title and summary, listing
   in the docs index, line budgets, filler words, issue numbers and figures.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REQUIRED_DOC_FILES = [
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "AGENTS.md",
    "docs/cli.md",
    "docs/internal/development.md",
    "docs/internal/releasing.md",
    "docs/models/README.md",
    "docs/server.md",
    "docs/internal/testing.md",
    "docs/internal/benchmarks.md",
    "docs/internal/performance.md",
    # BEGIN GENERATED MODEL DOCS
    *[f"docs/models/{model}.md"
      for model in ("deepseek-v4-flash", "embeddinggemma-2", "minimax-h3",
                    "qwen-image-2.1", "qwen3-asr", "qwen3-tts",
                    "qwen3.8-27b", "qwen3.8-flash-next")],
    *[f"docs/internal/models/{model}/{document}.md"
      for model in ("deepseek-v4-flash", "embeddinggemma-2", "minimax-h3",
                    "qwen-image-2.1", "qwen3-asr", "qwen3-tts",
                    "qwen3.8-27b", "qwen3.8-flash-next")
      for document in ("BENCHMARKS", "QUALITY", "EXPERIMENTS")],
    # END GENERATED MODEL DOCS
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
    # GitHub slugs the rendered text: drop a closing ATX sequence, link targets,
    # HTML tags and emphasis markers, but keep underscores inside words and code.
    t = re.sub(r'\s+#+\s*$', '', title)
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)
    t = re.sub(r'<[^>]+>', '', t)
    pieces = re.split(r'`([^`]*)`', t)
    t = "".join(piece if index % 2 else re.sub(r'(?<!\w)[*_]+|[*_]+(?!\w)', '', piece)
                for index, piece in enumerate(pieces))
    # Lowercase, remove punctuation other than '-' and '_', then map each
    # space to '-' without collapsing repeats.
    slug = re.sub(r'[^\w\- ]', '', t.strip().lower())
    return slug.replace(" ", "-")


def strip_fenced_blocks(content: str) -> str:
    """Blank fenced code blocks while preserving diagnostic line numbers."""
    lines = content.splitlines()
    fence = None
    for index, line in enumerate(lines):
        match = re.match(r' {0,3}(`{3,}|~{3,})', line)
        if fence is None:
            if match:
                marker = match.group(1)
                # Backtick fence info strings cannot contain backticks.
                if marker[0] == "`" and "`" in line[match.end():]:
                    continue
                fence = marker
                lines[index] = ""
        else:
            lines[index] = ""
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) \
                    and not line.strip().lstrip(fence[0]):
                fence = None
    return "\n".join(lines)


def extract_anchors(content: str) -> Set[str]:
    """Extract all heading anchor slugs, numbering repeats as GitHub does."""
    anchors: Set[str] = set()
    seen: Dict[str, int] = {}
    for match in MD_HEADING_REGEX.finditer(strip_fenced_blocks(content)):
        slug = slugify_heading(match.group("title").strip())
        if not slug:
            continue
        base_slug = slug
        count = seen.get(base_slug, 0)
        # Generated suffixes can collide with literal headings such as "Dup-1".
        while slug in anchors:
            count += 1
            slug = f"{base_slug}-{count}"
        seen[base_slug] = count
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

        # Code examples contain literal link syntax, not rendered links.
        lines = strip_fenced_blocks(content).splitlines()
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
                    # GitHub resolves '/path' from the repository root and
                    # everything else from the linking file's directory.
                    base = repo_root if file_part.startswith("/") else md_file.parent
                    resolved_target = (base / file_part.lstrip("/")).resolve()
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


# Public pages: docs/*.md and docs/models/*.md; docs/AGENTS.md and
# docs/internal/ are not published. See docs/AGENTS.md.
DOCS_INDEX = "docs/README.md"
MODELS_INDEX = "docs/models/README.md"
PAGE_BUDGET = 250
MODEL_PAGE_BUDGET = 120
FILLER_REGEX = re.compile(
    r"\b(simply|just|easily|basically|obviously|note that|it is worth|seamless(?:ly)?|blazing)\b",
    re.IGNORECASE)
ISSUE_REGEX = re.compile(r"(?<![\w&/#])#\d{2,}\b|github\.com/[^/\s]+/[^/\s]+/(?:pull|issues)/\d+")
FIGURE_DIR = "docs/figures"
FIGURE_STYLE = "docs/figures/style.css"
FIGURE_MAX_BYTES = 30 * 1024
FIGURE_FORBIDDEN = ("<script", "<foreignObject", "<image", "@import", "url(http", "@font-face")


def public_pages(repo_root: Path) -> List[Path]:
    """Published pages: docs/*.md and docs/models/*.md, without agent files."""
    pages = sorted((repo_root / "docs").glob("*.md")) + sorted((repo_root / "docs/models").glob("*.md"))
    return [p for p in pages if p.name not in {"AGENTS.md", "CLAUDE.md"}]


def prose_lines(content: str) -> List[str]:
    """Lines outside fenced code, with inline code and link targets blanked."""
    lines = []
    for line in strip_fenced_blocks(content).splitlines():
        line = re.sub(r"`[^`]*`", "``", line)
        line = re.sub(r"\]\([^)]*\)", "]()", line)
        lines.append(line)
    return lines


def check_public_pages(repo_root: Path) -> List[str]:
    """Apply the docs/AGENTS.md rules to every published page."""
    errors: List[str] = []
    index_links: Set[Path] = set()
    for index in (DOCS_INDEX, MODELS_INDEX):
        path = repo_root / index
        if path.exists():
            for match in MD_LINK_REGEX.finditer(strip_fenced_blocks(path.read_text(encoding="utf-8"))):
                target = match.group("target").split("#", 1)[0]
                if target and "://" not in target:
                    index_links.add((path.parent / target).resolve())
    for page in public_pages(repo_root):
        rel = page.relative_to(repo_root).as_posix()
        content = page.read_text(encoding="utf-8")
        lines = content.splitlines()
        if not lines or not lines[0].startswith("# "):
            errors.append(f"{rel}:1: Public page must start with a '# Title' line")
        elif len(lines) < 3 or lines[1] != "" or not lines[2].strip() or lines[2].startswith(("#", "|", "- ", "* ", "```", "~~~", "!", ">")):
            errors.append(f"{rel}:3: Title must be followed by a blank line and a summary paragraph")
        if rel not in (DOCS_INDEX, MODELS_INDEX) and page.resolve() not in index_links:
            errors.append(f"{rel}: Public page is not linked from {DOCS_INDEX} or {MODELS_INDEX}")
        budget = MODEL_PAGE_BUDGET if page.parent.name == "models" and page.name != "README.md" else PAGE_BUDGET
        if len(lines) > budget:
            errors.append(f"{rel}: {len(lines)} lines exceed the {budget}-line budget; split it or move detail to docs/internal/")
        for number, line in enumerate(prose_lines(content), start=1):
            for match in FILLER_REGEX.finditer(line):
                errors.append(f"{rel}:{number}: Filler word '{match.group(0)}'; see docs/AGENTS.md")
            for match in ISSUE_REGEX.finditer(line):
                errors.append(f"{rel}:{number}: Issue or pull request reference '{match.group(0)}' on a public page")
    return errors


def check_figures(repo_root: Path) -> List[str]:
    """SVG figures embed the canonical style and stay GitHub- and website-safe."""
    errors: List[str] = []
    style_path = repo_root / FIGURE_STYLE
    figure_dir = repo_root / FIGURE_DIR
    if not figure_dir.exists():
        return errors
    style = style_path.read_text(encoding="utf-8") if style_path.exists() else None
    if style is None:
        errors.append(f"{FIGURE_STYLE}: Missing canonical figure style")
    for svg in sorted(figure_dir.rglob("*.svg")):
        rel = svg.relative_to(repo_root).as_posix()
        text = svg.read_text(encoding="utf-8")
        if svg.stat().st_size > FIGURE_MAX_BYTES:
            errors.append(f"{rel}: Figure is larger than {FIGURE_MAX_BYTES // 1024} KB")
        for token in FIGURE_FORBIDDEN:
            if token.lower() in text.lower():
                errors.append(f"{rel}: Figure contains forbidden '{token}'")
        if re.search(r"\shref=\"(?!#)|xlink:href=\"(?!#)", text):
            errors.append(f"{rel}: Figure references an external resource")
        if re.search(r"\sstyle=\"|\s(?:fill|stroke|font-family|font-size)=\"(?!none)", text):
            errors.append(f"{rel}: Figure uses inline presentation; use classes from {FIGURE_STYLE}")
        root = re.search(r"<svg\b[^>]*>", text)
        if not root:
            errors.append(f"{rel}: No <svg> element")
            continue
        attrs = dict(re.findall(r'([\w:-]+)="([^"]*)"', root.group(0)))
        view_box = attrs.get("viewBox", "").split()
        if len(view_box) != 4 or view_box[2] != attrs.get("width") or view_box[3] != attrs.get("height"):
            errors.append(f"{rel}: width and height must equal the viewBox size")
        elif view_box[2] != "760":
            errors.append(f"{rel}: Figures are 760 units wide")
        if attrs.get("role") != "img" or not re.search(r"<title\b[^>]*>[^<]+</title>", text):
            errors.append(f"{rel}: Figure needs role=\"img\" and a non-empty <title>")
        styles = re.findall(r"<style>\n(.*?)</style>", text, flags=re.S)
        if style is not None and styles != [style]:
            errors.append(f"{rel}: <style> must embed {FIGURE_STYLE} verbatim, once")
        prefix = svg.stem + "-"
        for figure_id in re.findall(r'\sid="([^"]+)"', text):
            if not figure_id.startswith(prefix):
                errors.append(f"{rel}: id '{figure_id}' must start with '{prefix}'")
        if not re.search(r'<rect\b[^>]*class="fig-bg"', text):
            errors.append(f"{rel}: Figure must draw a fig-bg background rectangle")
    return errors


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

    # 3. Public documentation rules and figures
    all_errors.extend(check_public_pages(repo_root))
    all_errors.extend(check_figures(repo_root))

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
