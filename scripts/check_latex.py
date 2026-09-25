import os
import re
import sys

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

docs = [
    'README.md',
    'SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md',
    'PROBLEM_STATEMENT_AND_INITIAL_PLAN.md',
    'ROUND1_README.md',
    'CLAUDE.md',
    'APP_STATUS_REPORT.md',
    'DEMO.md',
    'FEATURE_PARITY.md',
    'AUDIT.md',
    'FINAL_JUDGE_EVALUATION_REPORT.md',
    'FINAL_NUMBERS_FOR_PPT.md',
    'CODE_REALITY_REPORT.md'
]

latex_macros = [r'\approx', r'\frac', r'\lambda', r'\sigma', r'\omega', r'\Delta', r'\text{', r'\begin{']
errors = []

for doc in docs:
    doc_path = os.path.join(workspace_root, doc)
    if not os.path.exists(doc_path):
        continue
    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        for macro in latex_macros:
            if macro in line:
                if 'NEVER use raw LaTeX syntax' in line or 'LaTeX' in line:
                    continue
                errors.append((doc, i, f"LaTeX macro '{macro}' found: {line.strip()}"))
        
        dollar_matches = re.finditer(r'(?<!\\)\$(?!\$|[a-zA-Z_0-9:]+\b)(.+?)\$', line)
        for m in dollar_matches:
            match_str = m.group(0)
            if '$env:' in match_str or '$LOCALAPPDATA' in match_str or '$' not in match_str:
                continue
            errors.append((doc, i, f"Math dollar block '{match_str}' found: {line.strip()}"))
        
        if '$$' in line:
            errors.append((doc, i, f"Double dollar block found: {line.strip()}"))

if errors:
    print(f"Found {len(errors)} LaTeX occurrences:")
    for doc, line_num, desc in errors:
        print(f"  {doc}:{line_num}: {desc}")
    sys.exit(1)
else:
    print("[SUCCESS] Zero LaTeX syntax found across all edited documents.")
    sys.exit(0)
