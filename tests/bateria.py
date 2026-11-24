"""Evalúa el clasificador con data/bateria_prompts.json (directos, benignos y evasiones).

Ejecutar:  python -m tests.bateria            python -m tests.bateria --csv
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
from src.chatbot import AsistenteSeguridad

RAIZ = Path(__file__).resolve().parent.parent


def ejecutar(modelo_texto="auto", umbral: float | None = None) -> list[dict]:
    """modelo_texto: "auto" (híbrido si existe el .joblib), None (solo reglas) o un pipeline."""
    bot = AsistenteSeguridad(modelo_texto=modelo_texto, **({} if umbral is None else {"umbral": umbral}))
    filas = []
    for item in json.loads((RAIZ / "data" / "bateria_prompts.json").read_text(encoding="utf-8")):
        bot.reiniciar()
        obtenida = bot.responder("Analiza este prompt: " + item["prompt"]).categoria
        filas.append({**item, "obtenida": obtenida, "ok": obtenida == item["categoria"],
                      "ok_binario": (obtenida is not None) == (item["categoria"] is not None)})
    return filas


def resumen(filas: list[dict]) -> dict:
    """`aciertos` = categoría exacta; `aciertos_binarios` = ataque/no ataque (comparable con el ML puro)."""
    por_tipo = defaultdict(lambda: [0, 0, 0])
    for f in filas:
        por_tipo[f["tipo"]][0] += f["ok"]
        por_tipo[f["tipo"]][1] += 1
        por_tipo[f["tipo"]][2] += f["ok_binario"]
    return {t: {"aciertos": a, "total": n, "tasa": round(a / n, 3), "aciertos_binarios": b} for t, (a, n, b) in por_tipo.items()}


def comparar() -> dict:
    """Reglas (Avance 1) vs híbrido (ML + reglas) por tipo, y guarda los errores para el análisis."""
    reglas, hibrido = ejecutar(None), ejecutar("auto")
    ruta = RAIZ / "evidencias" / "final"; ruta.mkdir(parents=True, exist_ok=True)
    with open(ruta / "bateria_reglas_vs_ml.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["tipo", "categoria_esperada", "reglas", "hibrido", "ok_reglas", "ok_hibrido", "prompt"])
        for a, b in zip(reglas, hibrido):
            w.writerow([a["tipo"], a["categoria"], a["obtenida"], b["obtenida"], int(a["ok"]), int(b["ok"]), a["prompt"]])
    salida = {"reglas": resumen(reglas), "hibrido": resumen(hibrido)}
    (ruta / "bateria_reglas_vs_ml.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    return salida


def barrido_umbral(umbrales=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)) -> list[dict]:
    """Híbrido por umbral. Solo para análisis de sensibilidad: el umbral NO se ajusta sobre la batería de evaluación."""
    ruta = RAIZ / "evidencias" / "final"; ruta.mkdir(parents=True, exist_ok=True)
    filas = []
    for u in umbrales:
        r = resumen(ejecutar("auto", u))
        filas.append({"umbral": u, **{f"{t}_{k}": r[t][k] for t in r for k in ("aciertos", "aciertos_binarios")}})
    with open(ruta / "bateria_umbral.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    return filas


if __name__ == "__main__":
    if "--barrido" in sys.argv:
        for f in barrido_umbral(): print(f)
        raise SystemExit
    if "--comparar" in sys.argv:
        print(json.dumps(comparar(), ensure_ascii=False, indent=2)); raise SystemExit
    filas = ejecutar(None if "--reglas" in sys.argv else "auto")
    for f in filas:
        print(f"[{'OK ' if f['ok'] else 'ERR'}] {f['tipo']:8} esperada={f['categoria']!s:22} obtenida={f['obtenida']!s:22} | {f['prompt'][:70]}")
    r = resumen(filas)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if "--csv" in sys.argv:
        ruta = RAIZ / "evidencias"; ruta.mkdir(exist_ok=True)
        with open(ruta / "bateria_resultados.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["tipo", "categoria", "obtenida", "ok", "prompt"]); w.writeheader(); w.writerows(filas)
        (ruta / "bateria_resumen.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
