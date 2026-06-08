import os
import re

regex = re.compile(
    r"matplotlib|pyplot|plt\.|savefig|Figure|Axes3D|plot\(|scatter\(|imshow\(|contour|pcolormesh|plot_surface|includegraphics|\.png|\.pdf",
    re.IGNORECASE
)

root_dir = r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order"
ignore_dirs = {".venv", ".git", ".pytest_cache", ".pytest_tmp", "__pycache__", "build", "dist", ".benchmarks", ".runtime_cache", ".runtime_native"}
ignore_extensions = {".png", ".pdf", ".jpg", ".jpeg", ".gif", ".pyc", ".pyd", ".synctex.gz", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".toc"}

matches = []

for root, dirs, files in os.walk(root_dir):
    # Prune ignored directories
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in ignore_extensions:
            continue
        filepath = os.path.join(root, file)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    if regex.search(line):
                        rel_path = os.path.relpath(filepath, root_dir)
                        matches.append((rel_path, line_idx, line.strip()))
        except Exception as e:
            # Skip unreadable files
            pass

# Output matches to a text file for our reference
output_file = os.path.join(root_dir, "scratch", "plotting_references.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for rel_path, line_idx, content in matches:
        f.write(f"{rel_path}:{line_idx}:{content}\n")

print(f"Search complete. Found {len(matches)} occurrences. Output written to {output_file}")
