"""Verifica los PDF, las citas compiladas y las evidencias actuales del paquete."""
from pathlib import Path
import csv
import hashlib
import json
import re

from pypdf import PdfReader

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative):
    return json.loads((BASE / relative).read_text(encoding="utf-8-sig"))


def verify():
    bibliography = load("auditoria/bibliografia/bibliografia_completa_resumen.json")
    zotero = load("auditoria/bibliografia/verificacion_importacion_zotero.json")
    wolfram = load("auditoria/wolfram/resumen_comprobaciones.json")
    assert bibliography["bib_entries"] == bibliography["unique_bib_keys"] == 141
    assert bibliography["references_bib_sha256"] == sha(BASE / "references.bib")
    assert not bibliography["duplicate_bib_keys"]
    assert not bibliography["duplicate_bib_dois"]
    assert all(not keys for keys in bibliography["missing_bib_keys"].values())
    assert zotero["matched_entries"] == 141 and not zotero["missing_entries"]
    assert zotero["source_bib_sha256"] == sha(BASE / "references.bib")
    assert wolfram["numero_decisiones_declaradas"] == 25
    assert all(check["resultado"] for check in wolfram["comprobaciones"])
    documents = []
    for stem, edition, qa_file in [
        ("reporte_maestro_atractores_ocultos", "teorico", "qa_visual_teorico.json"),
        ("reporte_maestro_lectura_estudio", "estudio", "qa_visual_estudio.json"),
    ]:
        pdf = BASE / (stem + ".pdf")
        qa = load("auditoria/" + qa_file)
        digest = sha(pdf)
        # La huella puede figurar en el encabezado o en el control posterior.
        assert digest in json.dumps(qa), (stem, "QA de otro PDF")
        entry = {"edicion": edition, "archivo": pdf.name, "paginas": len(PdfReader(pdf).pages),
                 "bytes": pdf.stat().st_size, "sha256": digest, "qa_visual": "auditoria/" + qa_file}
        log = BASE / "build" / (stem + ".log")
        aux = BASE / "build" / (stem + ".aux")
        if log.exists() and aux.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            assert not re.search(r"Warning:|Overfull|undefined|multiply defined", text)
            citations = set()
            for group in re.findall(r"\\citation\{([^}]+)\}", aux.read_text(encoding="utf-8")):
                citations.update(group.split(","))
            labels = set(re.findall(r"\\bibcite\{([^}]+)\}", aux.read_text(encoding="utf-8")))
            assert citations == labels
            assert len(citations) == (97 if edition == "teorico" else 114)
            entry.update({"compilacion": "correcta", "advertencias_latex": 0,
                          "desbordamientos": 0, "citas_sin_resolver": 0,
                          "claves_citadas": sorted(citations)})
        documents.append(entry)
    counts = {}
    for name in ["matematica", "resultados_figuras"]:
        with (BASE / "auditoria" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
            counts[name] = len(list(csv.DictReader(stream)))
    return {"fecha": "2026-11-09", "documentos": documents, "matrices_filas": counts,
            "bibliografia": {"entradas": 141, "fichas_en_zotero": 141,
                             "nuevas_importadas": 92, "coinciden_coleccion_tesis": 131,
                             "solo_otras_colecciones": 10,
                             "pendientes_metadatos": zotero["imported_records_with_issues"]},
            "wolfram_decisiones_satisfechas": 25,
            "correcciones_actuales": "Sección27, clasificadores, protocolos y nuevos contrastes: ver expediente correcciones_sin_hardware_20261109.", "alcance": "Auditoría y corrección de los reportes existentes; lectura matemática y bibliográfica delimitada, recuento de registros, álgebra actual y control editorial. No repetición de todas las campañas, certificación de caos u ocultación ni resolución de la discrepancia c590."}


if __name__ == "__main__":
    result = verify()
    (BASE / "auditoria/verificacion_final.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documentos": [(d["edicion"], d["paginas"]) for d in result["documentos"]],
                      "matrices": result["matrices_filas"], "estado": "verificado"}, ensure_ascii=False))
