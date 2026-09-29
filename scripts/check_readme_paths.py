#!/usr/bin/env python3
"""
Check all local file paths and internal #anchor links referenced in README.md.
Fails if any referenced local path does not exist or if any #anchor does not match
a GitHub-compatible heading slug in README.md.
"""

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"


def get_github_heading_slug(heading: str) -> str:
    """Generate GitHub markdown heading slug."""
    # Strip HTML tags
    h = re.sub(r"<[^>]+>", "", heading)
    # Strip markdown emphasis/code backticks
    h = re.sub(r"[`\*_]", "", h)
    # Remove punctuation except letters, digits, spaces, and hyphens/underscores
    h = re.sub(r"[^\w\s-]", "", h.lower(), flags=re.UNICODE).strip()
    # Replace spaces with hyphens
    slug = re.sub(r"[-\s]+", "-", h)
    return slug


def extract_headings(text: str):
    """Extract all markdown headings and return a mapping of slug -> heading text."""
    slug_counts = {}
    slugs = set()
    heading_list = []
    
    # Track code fences so we don't treat comments inside code as headings
    in_code_fence = False
    
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
            
        m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if m:
            heading_text = m.group(2).strip()
            base_slug = get_github_heading_slug(heading_text)
            count = slug_counts.get(base_slug, 0)
            slug_counts[base_slug] = count + 1
            final_slug = base_slug if count == 0 else f"{base_slug}-{count}"
            slugs.add(final_slug)
            heading_list.append((final_slug, heading_text))
            
    return slugs, heading_list


def check_readme():
    if not README_PATH.exists():
        print(f"ERROR: {README_PATH} not found.")
        sys.exit(1)

    with open(README_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Extract headings and generate slugs
    valid_slugs, heading_list = extract_headings(text)

    # 2. Extract links and sources
    # Exclude code fences from link extraction
    content_lines = []
    in_code_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if not in_code_fence:
            content_lines.append(line)
    clean_text = "\n".join(content_lines)

    # Patterns
    # Markdown links (supporting nested badge images [![alt](img)](target) as well as regular [text](target))
    md_links = re.findall(r"\[(?:\!\[[^\]]*\]\([^)]+\)|[^\]])*\]\(([^)]+)\)", clean_text)
    # Also find any remaining simple links
    simple_links = re.findall(r"\]\(([^)]+)\)", clean_text)
    # HTML src: src="target"
    html_src = re.findall(r'src=["\']([^"\']+)["\']', clean_text)
    # HTML href: href="target"
    html_href = re.findall(r'href=["\']([^"\']+)["\']', clean_text)

    all_raw_targets = md_links + simple_links + html_src + html_href

    local_paths = []
    anchor_links = []

    for target in all_raw_targets:
        target = target.strip()
        if not target:
            continue
        # Check if external URL or mailto
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        # Check if pure anchor link
        if target.startswith("#"):
            anchor_links.append(target[1:])
            continue
        # Split target into path and possible anchor (e.g. file.md#section)
        if "#" in target:
            path_part, anchor_part = target.split("#", 1)
            if path_part:
                local_paths.append(path_part)
        else:
            local_paths.append(target)

    # Unique while preserving order
    unique_local_paths = list(dict.fromkeys(local_paths))
    unique_anchor_links = list(dict.fromkeys(anchor_links))

    print(f"=== Checking README.md Paths and Anchors ===")
    print(f"Found {len(unique_local_paths)} unique local paths.")
    print(f"Found {len(unique_anchor_links)} unique anchor links.")
    print(f"Extracted {len(valid_slugs)} heading slugs.")

    # 3. Check local paths
    missing_paths = []
    print("\n--- Local File Paths ---")
    for path_str in unique_local_paths:
        # Resolve relative to repo root
        resolved = REPO_ROOT / path_str
        status = "OK" if resolved.exists() else "MISSING"
        if not resolved.exists():
            missing_paths.append(path_str)
        print(f"[{status}] {path_str}")

    # 4. Check anchors
    missing_anchors = []
    print("\n--- Anchor Links ---")
    for anchor in unique_anchor_links:
        status = "OK" if anchor in valid_slugs else "INVALID"
        if anchor not in valid_slugs:
            missing_anchors.append(anchor)
        print(f"[{status}] #{anchor}")

    # Summary
    print("\n=== Summary ===")
    has_error = False
    if missing_paths:
        print(f"FAILED: {len(missing_paths)} missing file path(s):")
        for p in missing_paths:
            print(f"  - {p}")
        has_error = True
    else:
        print("PASS: All local file paths exist.")

    if missing_anchors:
        print(f"FAILED: {len(missing_anchors)} invalid anchor link(s):")
        for a in missing_anchors:
            print(f"  - #{a}")
        has_error = True
    else:
        print("PASS: All anchor links match heading slugs.")

    if has_error:
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    check_readme()
