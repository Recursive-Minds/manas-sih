import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.run_final_benchmark import sync_readme, sync_architecture_doc

report_path = os.path.join(ROOT_DIR, "FINAL_JUDGE_EVALUATION_REPORT.md")
if not os.path.exists(report_path):
    print(f"Error: {report_path} not found.")
    sys.exit(1)

with open(report_path, "r", encoding="utf-8") as f:
    md_content = f.read()

print("Synchronizing documentation across the workspace...")
sync_readme(md_content)
sync_architecture_doc(md_content)

# Also fix any relative image paths in SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md
arch_path = os.path.join(ROOT_DIR, "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md")
if os.path.exists(arch_path):
    with open(arch_path, "r", encoding="utf-8") as f:
        arch_text = f.read()
    arch_fixed = arch_text.replace("../artifacts/", "artifacts/")
    with open(arch_path, "w", encoding="utf-8") as f:
        f.write(arch_fixed)
    print("Fixed relative image paths in SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md.")

print("All documents synchronized successfully.")
