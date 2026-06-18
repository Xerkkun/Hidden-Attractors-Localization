import re
from pathlib import Path

def test_latex_figure_paths():
    root_dir = Path(__file__).resolve().parents[2]
    
    tex_files = [
        root_dir / "version_2/docs/reporte_unificado_chua_fraccionario.tex"
    ]
    
    for tf in tex_files:
        if not tf.exists():
            continue
            
        content = tf.read_text(encoding="utf-8", errors="ignore")
        
        # 1. Scan for \includegraphics
        pattern = r"\\includegraphics(\[[^\]]*\])?\{([^\}]+)\}"
        matches = re.findall(pattern, content)
        
        for options, path in matches:
            path = path.strip()
            if path == "inaoe.png" or "#2" in path:
                continue  # skip institutional logo and macro parameter definition
                
            # Verify it references library_figures and has .pdf extension
            assert "library_figures" in path, f"Graphics path '{path}' in {tf.name} does not reference 'library_figures'."
            assert path.endswith(".pdf"), f"Graphics path '{path}' in {tf.name} must be a PDF file."
            
        # 2. Scan for \maybeincludegraphics
        pattern_calls = r"\\maybeincludegraphics(\[[^\]]*\])?\{([^\}]+)\}"
        matches_calls = re.findall(pattern_calls, content)
        
        for options, filename in matches_calls:
            filename = filename.strip()
            assert filename.endswith(".pdf"), f"maybeincludegraphics filename '{filename}' in {tf.name} must be a PDF file."
