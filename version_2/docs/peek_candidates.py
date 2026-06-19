import os

docs_dir = r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2\docs"
candidates = [
    "adapting_new_systems.md",
    "cli_migration_legacy_entrypoints.md",
    "contributing.md",
    "examples.md",
    "examples_index.md",
    "external_tools.md",
    "getting_started.md",
    "installation.md",
    "migration_unified_methodology.md",
    "notebooks.md",
    "quick_start.md",
    "scientific_scope.md",
    "systems.md",
    "testing.md",
    "workflows.md"
]

for filename in candidates:
    filepath = os.path.join(docs_dir, filename)
    if os.path.exists(filepath):
        print("="*60)
        print(f"FILE: {filename}")
        print("="*60)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for line in lines[:25]:
            print(line.rstrip())
