import os
from pathlib import Path

# Paths
ROOT = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order")
MAIN_TEX_PATH = ROOT / "DF y NC Chua entero y fraccionario copy" / "main.tex"
NEW_RESULTS_PATH = ROOT / "scratch" / "new_results.tex"

if not MAIN_TEX_PATH.exists():
    print(f"Error: main.tex not found at {MAIN_TEX_PATH}")
    exit(1)

if not NEW_RESULTS_PATH.exists():
    print(f"Error: new_results.tex not found at {NEW_RESULTS_PATH}")
    exit(1)

# Read main.tex lines
with open(MAIN_TEX_PATH, "r", encoding="utf-8") as f:
    main_lines = f.readlines()

# Read new results
with open(NEW_RESULTS_PATH, "r", encoding="utf-8") as f:
    new_results_content = f.read()

# Find key lines
idx_results = -1
idx_alcance = -1
idx_danca_analysis = -1

for idx, line in enumerate(main_lines):
    if "\\section{Resultados numéricos}" in line and idx_results == -1:
        idx_results = idx
    elif "\\section{Alcance de la metodología y sistemas candidatos}" in line and idx_alcance == -1:
        idx_alcance = idx
    elif "\\section{Análisis crítico de Danca (2017) y verificación de ocultedad}" in line and idx_danca_analysis == -1:
        idx_danca_analysis = idx

print(f"Found sections at lines:")
print(f"  Resultados: {idx_results + 1}")
print(f"  Alcance: {idx_alcance + 1}")
print(f"  Danca Analysis: {idx_danca_analysis + 1}")

if idx_results == -1 or idx_alcance == -1 or idx_danca_analysis == -1:
    print("Error: Could not find all section indices.")
    exit(1)

# Build the new main.tex content
# Part 1: Everything before \section{Resultados numéricos}
part1 = "".join(main_lines[:idx_results])

# Part 2: The new results section
part2 = new_results_content + "\n"

# Part 3: The remaining sections from Alcance to Conclusiones
# We read from idx_alcance up to idx_danca_analysis (exclusive)
part3 = "".join(main_lines[idx_alcance:idx_danca_analysis])

# Part 4: Bibliography and end document
part4 = """
\\bibliographystyle{IEEEtran}
\\bibliography{phd-bibliography}

\\end{document}
"""

final_content = part1 + part2 + part3 + part4

# Write back to main.tex
with open(MAIN_TEX_PATH, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Merged and updated main.tex successfully!")
