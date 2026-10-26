"""Corre las 13 conversaciones y la batería, y guarda las evidencias en evidencias/.

Ejecutar:  python run_evidencias.py
Genera: conversaciones.json, conversaciones_transcripcion.txt,
        bateria_resultados.csv y bateria_resumen.json
"""
import contextlib
import io
import json
from pathlib import Path

from tests import bateria, conversaciones

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "evidencias"


def main() -> None:
    SALIDA.mkdir(exist_ok=True)

    res = conversaciones.ejecutar()
    (SALIDA / "conversaciones.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        conversaciones.imprimir(res)
    (SALIDA / "conversaciones_transcripcion.txt").write_text(buffer.getvalue(), encoding="utf-8")

    filas = bateria.ejecutar()
    resumen = bateria.resumen(filas)
    import csv
    with open(SALIDA / "bateria_resultados.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tipo", "categoria", "obtenida", "ok", "prompt"], extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    (SALIDA / "bateria_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_c = sum(c["ok"] for c in res)
    print(f"Conversaciones: {ok_c}/{len(res)}")
    for tipo, r in resumen.items():
        print(f"Batería {tipo}: {r['aciertos']}/{r['total']} ({r['tasa']:.0%})")
    print(f"Evidencias guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
