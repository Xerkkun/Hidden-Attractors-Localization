# -*- coding: utf-8 -*-
import pytest
import re
from pathlib import Path
from tests.helpers.test_documentation_text import read, active_doc_paths, get_violations_without_context, ROOT

DEPRECATION_KEYWORDS = [
    "legacy", "deprecated", "removed", "no longer installed", "not public",
    "migration", "obsolete", "obsoleto", "deprecado", "no público",
    "no se instala", "reemplazado", "antiguo", "históricos", "migración"
]

LEGACY_COMMANDS = [
    "hidden-attractors-check-validation",
    "hidden-attractors-protocol",
    "hidden-attractors-sphere-controls",
    "hidden-attractors-refined-basin",
    "hidden-attractors-robustness-overlay",
    "hidden-attractors-danca-abm-sphere-controls",
    "hidden-attractors-fractional-report-run",
]

RECOMMENDED_PHRASES = [
    "use hidden-attractors-check-validation",
    "run hidden-attractors-check-validation",
    "execute hidden-attractors-protocol",
    "use hidden-attractors-protocol",
]

def check_code_blocks(content: str, filename: str) -> list[str]:
    violations = []
    # Find all code blocks with optional language specifier
    for match in re.finditer(r"```(bash|sh|shell|powershell|text|cmd)?\n(.*?)```", content, re.DOTALL | re.IGNORECASE):
        block_text = match.group(2)
        has_legacy_cmd = any(cmd in block_text for cmd in LEGACY_COMMANDS)
        if has_legacy_cmd:
            block_start = match.start()
            # Find the nearest preceding header line
            last_header = ""
            pos = content.rfind("\n#", 0, block_start)
            if pos != -1:
                end_line = content.find("\n", pos + 1)
                if end_line == -1:
                    end_line = len(content)
                last_header = content[pos:end_line].strip()
                
            deprecation_headings = [
                "legacy", "migration", "deprecated", "obsolete", "comandos obsoletos", "migración"
            ]
            is_deprecated_section = any(h.lower() in last_header.lower() for h in deprecation_headings)
            if not is_deprecated_section:
                for cmd in LEGACY_COMMANDS:
                    if cmd in block_text:
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append(
                            f"L{line_num}: Code block under non-deprecation header '{last_header}' contains legacy command '{cmd}'"
                        )
    return violations

@pytest.mark.hygiene
def test_no_legacy_cli_in_manuals_blocks_and_phrases():
    docs = active_doc_paths()
    violations = []
    
    for p in docs:
        content = read(p)
        
        # 1. Check code blocks
        block_errs = check_code_blocks(content, p.name)
        for err in block_errs:
            violations.append(f"{p.name} -> {err}")
            
        # 2. Check forbidden recommended phrases without deprecation context
        for phrase in RECOMMENDED_PHRASES:
            if phrase in content.lower():
                phrase_errs = get_violations_without_context(content, phrase, DEPRECATION_KEYWORDS, window=160)
                for err in phrase_errs:
                    violations.append(f"{p.name}:{err} for recommendation phrase '{phrase}'")
                    
    assert not violations, (
        "Legacy CLI commands recommended in executable context or active blocks:\n"
        + "\n".join(violations)
    )
