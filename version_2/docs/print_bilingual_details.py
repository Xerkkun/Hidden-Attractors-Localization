import os

docs_dir = r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2\docs"
files_to_print = [
    "installation.md",
    "testing.md",
    "external_tools.md",
    "adapting_new_systems.md"
]

for filename in files_to_print:
    filepath = os.path.join(docs_dir, filename)
    if os.path.exists(filepath):
        print("="*80)
        print(f"FILE: {filename}")
        print("="*80)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            print(f.read())
