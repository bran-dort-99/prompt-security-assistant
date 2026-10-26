"""Evalúa el clasificador con data/bateria_prompts.json (directos, benignos y evasiones).

Ejecutar:  python -m tests.bateria            python -m tests.bateria --csv
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
from src.chatbot import AsistenteSeguridad

RAIZ = Path(__file__).resolve().parent.parent


def ejecutar() -> list[dict]:
    bot = AsistenteSeguridad()
    filas = []
    for item in json.loads((RAIZ / "data" / "bateria_prompts.json").read_text(encoding="utf-8")):
        bot.reiniciar()
        obtenida = bot.responder("Analiza este prompt: " + item["prompt"]).categoria
        filas.append({**item, "obtenida": obtenida, "ok": obtenida == item["categoria"]})
    return filas


def resumen(filas: list[dict]) -> dict:
    por_tipo = defaultdict(lambda: [0, 0])
    for f in filas:
        por_tipo[f["tipo"]][0] += f["ok"]
        por_tipo[f["tipo"]][1] += 1
    return {t: {"aciertos": a, "total": n, "tasa": round(a / n, 3)} for t, (a, n) in por_tipo.items()}


if __name__ == "__main__":
    filas = ejecutar()
    for f in filas:
        print(f"[{'OK ' if f['ok'] else 'ERR'}] {f['tipo']:8} esperada={f['categoria']!s:22} obtenida={f['obtenida']!s:22} | {f['prompt'][:70]}")
    r = resumen(filas)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if "--csv" in sys.argv:
        ruta = RAIZ / "evidencias"; ruta.mkdir(exist_ok=True)
        with open(ruta / "bateria_resultados.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["tipo", "categoria", "obtenida", "ok", "prompt"]); w.writeheader(); w.writerows(filas)
        (ruta / "bateria_resumen.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
