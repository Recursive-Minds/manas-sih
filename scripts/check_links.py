import os
import re
import sys

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

docs = [
    'README.md',
    'SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md',
    'PROBLEM_STATEMENT_AND_INITIAL_PLAN.md',
    'FINAL_JUDGE_EVALUATION_REPORT.md',
    'FINAL_NUMBERS_FOR_PPT.md',
    'CODE_REALITY_REPORT.md'
]

link_pattern = re.compile(r'!?\[([^\]]*)\]\(([^)]+)\)')
img_tag_pattern = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']')

def slugify(h):
    h = h.strip().lower()
    # Remove markdown link markup if any
    h = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', h)
    # Remove code, bold, italic
    h = re.sub(r'[`*_]', '', h)
    # Remove HTML tags
    h = re.sub(r'<[^>]+>', '', h)
    # Remove punctuation except hyphen and space
    h = re.sub(r'[^\w\s-]', '', h)
    # Convert spaces to hyphens (preserving multiple spaces as multiple hyphens per GitHub)
    h = h.replace(' ', '-')
    return h.strip('-')

errors = []
total_checked = 0

for doc in docs:
    doc_path = os.path.join(workspace_root, doc)
    if not os.path.exists(doc_path):
        continue
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Build heading slugs for anchor verification
    headings = re.findall(r'^(#{1,6})\s+(.+)$', content, flags=re.MULTILINE)
    slugs = set()
    slug_counts = {}
    for level, h in headings:
        slug = slugify(h)
        if slug in slug_counts:
            slug_counts[slug] += 1
            slug = f"{slug}-{slug_counts[slug]}"
        else:
            slug_counts[slug] = 0
        slugs.add(slug)

    # Check markdown links
    for match in link_pattern.finditer(content):
        url = match.group(2).strip()
        if '...' in url or '<' in url or 'XYZ' in url or 'file_path' in url:
            continue
        if url.startswith('http://') or url.startswith('https://') or url.startswith('mailto:'):
            continue

        # In-page anchor check
        if url.startswith('#'):
            anchor = url[1:].strip()
            total_checked += 1
            if anchor and anchor not in slugs:
                errors.append((doc, f"Broken in-page anchor: {url}"))
            continue

        # file:/// link
        if url.startswith('file:///'):
            path = url.replace('file:///', '')
            path = path.replace('/', os.sep)
            path = path.split('#')[0]
            total_checked += 1
            if not os.path.exists(path):
                errors.append((doc, f"Broken file:/// link: {url} -> {path}"))
            continue

        # relative path
        rel_path = url.split('#')[0]
        if not rel_path:
            continue
        rel_path = rel_path.replace('/', os.sep)
        full_path = os.path.normpath(os.path.join(os.path.dirname(doc_path), rel_path))
        total_checked += 1
        if not os.path.exists(full_path):
            errors.append((doc, f"Broken relative link: {url} -> {full_path}"))

    # Check img tags
    for match in img_tag_pattern.finditer(content):
        src = match.group(1).strip()
        if src.startswith('http://') or src.startswith('https://') or '...' in src or '<' in src:
            continue
        src_path = src.replace('/', os.sep)
        full_path = os.path.normpath(os.path.join(os.path.dirname(doc_path), src_path))
        total_checked += 1
        if not os.path.exists(full_path):
            errors.append((doc, f"Broken img src: {src} -> {full_path}"))

print(f"Total links/anchors checked: {total_checked}")
if errors:
    print(f"Found {len(errors)} link/anchor errors:")
    for doc, err in errors:
        print(f"  [{doc}] {err}")
    sys.exit(1)
else:
    print("[SUCCESS] All links and in-page anchors are valid!")
    sys.exit(0)
