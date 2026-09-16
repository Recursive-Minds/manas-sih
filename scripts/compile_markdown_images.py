"""
compile_markdown_images.py
--------------------------
Maintains clean relative image links in Markdown documents so that documents
remain lightweight (<50KB) and load properly in GitHub and IDE Markdown previews
without exhausting server context or memory.
"""

import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def verify_documents():
    docs = [
        os.path.join(ROOT_DIR, "docs", "SYSTEM_IMPLEMENTATION_AND_TECHNIQUES_RECORD.md"),
        os.path.join(ROOT_DIR, "FINAL_JUDGE_EVALUATION_REPORT.md")
    ]
    for d in docs:
        if not os.path.exists(d):
            continue
        with open(d, "r", encoding="utf-8") as f:
            content = f.read()
        imgs = re.findall(r'<img [^>]+>', content)
        b64_imgs = [img for img in imgs if "data:image/png;base64," in img]
        size_kb = os.path.getsize(d) / 1024
        print(f"[{os.path.basename(d)}] Size: {size_kb:.1f} KB | Total images: {len(imgs)} | Base64 images: {len(b64_imgs)}")
        for idx, img in enumerate(imgs):
            alt_m = re.search(r'alt="([^"]+)"', img)
            src_m = re.search(r'src="([^"]+)"', img)
            alt = alt_m.group(1) if alt_m else "no-alt"
            src = src_m.group(1) if src_m else "no-src"
            is_b64 = "data:image/png;base64," in src
            print(f"   Image {idx+1}: {alt} -> src: {src[:60]}... (Base64: {is_b64})")

if __name__ == "__main__":
    verify_documents()
