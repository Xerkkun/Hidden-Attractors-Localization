import re
from pathlib import Path

root_dir = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order")

def update_main_tex(filepath):
    if not filepath.exists():
        print(f"Skipping {filepath}: not found.")
        return
        
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    
    # We want to replace any \includegraphics[...]{path} where path starts with Figs/ or is one of the target figures.
    # We will exclude logo image "inaoe.png".
    
    def replace_path(match):
        options = match.group(1) or ""
        path = match.group(2).strip()
        
        if path == "inaoe.png":
            return match.group(0)
            
        # Get filename stem
        stem = Path(path).stem
        
        # New relative path from DF y NC Chua.../ to library_figures:
        # .. / version_2 / library_figures / by_report / df_nc_chua / pdf / stem.pdf
        new_path = f"../version_2/library_figures/by_report/df_nc_chua/pdf/{stem}.pdf"
        
        return f"\\includegraphics{options}{{{new_path}}}"
        
    # Pattern to match \includegraphics[options]{path}
    pattern = r"\\includegraphics(\[[^\]]*\])?\{([^\}]+)\}"
    new_content = re.sub(pattern, replace_path, content)
    
    if new_content != content:
        filepath.write_text(new_content, encoding="utf-8")
        print(f"Updated LaTeX references in {filepath}")
    else:
        print(f"No changes made to {filepath}")

def update_unified_report(filepath):
    if not filepath.exists():
        print(f"Skipping {filepath}: not found.")
        return
        
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    
    # In reporte_unificado_chua_fraccionario.tex, we have a custom macro \maybeincludegraphics
    # Let's see lines 32-40:
    # \newcommand{\maybeincludegraphics}[2][]{%
    #     \includegraphics[#1]{assets/figures/chua_fractional_report/#2}%
    #     ...
    # We can rewrite the macro to load from the canonical library figures directory:
    # \includegraphics[#1]{../library_figures/by_report/unified_chua_fractional/pdf/#2}
    # And we also want to change the file extension from .png to .pdf inside the macro calls.
    
    # 1. Update the macro definition to use our unified directory
    target_macro = r"\\newcommand\{\\maybeincludegraphics\}\[2\]\[\]\{%\n    \\includegraphics\[#1\]\{assets/figures/chua_fractional_report/#2\}%"
    # Let's do a simpler replacement of the paths inside the macro
    content = content.replace("assets/figures/chua_fractional_report/", "../library_figures/by_report/unified_chua_fractional/pdf/")
    content = content.replace("assets/figures/chua_integer_q1/", "../library_figures/by_report/unified_chua_fractional/pdf/")
    content = content.replace("assets/figures/efork3_validation/", "../library_figures/by_report/unified_chua_fractional/pdf/")
    
    # 2. Inside the calls to \maybeincludegraphics, change the extension from .png to .pdf
    def replace_extension(match):
        options = match.group(1) or ""
        filename = match.group(2).strip()
        stem = Path(filename).stem
        return f"\\maybeincludegraphics{options}{{{stem}.pdf}}"
        
    pattern_calls = r"\\maybeincludegraphics(\[[^\]]*\])?\{([^\}]+)\}"
    new_content = re.sub(pattern_calls, replace_extension, content)
    
    if new_content != content:
        filepath.write_text(new_content, encoding="utf-8")
        print(f"Updated LaTeX references in {filepath}")
    else:
        print(f"No changes made to {filepath}")

update_main_tex(root_dir / "DF y NC Chua entero y fraccionario copy/main.tex")
update_main_tex(root_dir / "DF y NC Chua entero y fraccionario/main.tex")
update_unified_report(root_dir / "version_2/docs/reporte_unificado_chua_fraccionario.tex")
