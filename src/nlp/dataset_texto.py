"""Construye data/intenciones/dataset.csv (texto, etiqueta, fuente, idioma, split, plantilla_id).

- Siempre incluye los ejemplos propios (propios_es_en.csv).
- Si hay red y la librería `datasets` (Colab), suma corpus públicos (D8/plan §4.1).
- Excluye por texto normalizado: deepset TEST y la batería de 36 prompts del Avance 1.
- Partición 80/20 estratificada y agrupada por `plantilla_id` (StratifiedGroupKFold).

    python -m src.nlp.dataset_texto [--sin-red]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.nlp.preprocesamiento import normalizar

RAIZ = Path(__file__).resolve().parent.parent.parent
DIR = RAIZ / "data" / "intenciones"
PROPIOS = DIR / "propios_es_en.csv"
DATASET = DIR / "dataset.csv"
BATERIA = RAIZ / "data" / "bateria_prompts.json"
RESUMEN = RAIZ / "evidencias" / "avance2" / "dataset_resumen.csv"
SEMILLA = 42

ETIQUETAS = [
    "saludo", "despedida", "agradecimiento", "ayuda", "consulta_seguridad", "benigno",
    "prompt_injection", "indirect_injection", "jailbreak", "system_prompt_leakage", "data_leakage", "excessive_agency",
]
OBJETIVO_HF = {"prompt_injection": 300, "jailbreak": 300, "system_prompt_leakage": 150, "benigno": 400}

RE_LEAK_SISTEMA = re.compile(
    r"system prompt|instructions above|initial prompt|reveal your (rules|prompt|instructions)|"
    r"prompt del sistema|instrucciones (iniciales|anteriores|de arriba)|revela tus (reglas|instrucciones)|"
    r"tus instrucciones|your instructions|above instructions|print.*(prompt|instructions)",
    re.I,
)
RE_LEAK_DATOS = re.compile(r"password|contrase[nñ]a|api[ _-]?key|credential|credencial|secret|token|ssn|credit card|tarjeta", re.I)


def _clave(texto: str) -> str:
    return " ".join(normalizar(texto).split())


def cargar_propios() -> pd.DataFrame:
    df = pd.read_csv(PROPIOS)
    df["fuente"] = "propio"
    return df


def exclusiones(extra: set[str] | None = None) -> set[str]:
    claves = set(extra or ())
    if BATERIA.exists():
        for item in json.loads(BATERIA.read_text(encoding="utf-8")):
            claves.add(_clave(item["prompt"]))
    return claves


def _muestra(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=SEMILLA) if len(df) else df


def _fila(texto: str, etiqueta: str, fuente: str, idioma: str, i: int) -> dict:
    return {"texto": str(texto).strip(), "etiqueta": etiqueta, "idioma": idioma, "plantilla_id": f"{fuente}-{i}", "fuente": fuente}


def _reasignar_inyeccion(texto: str) -> str:
    if RE_LEAK_SISTEMA.search(texto):
        return "system_prompt_leakage"
    return "prompt_injection"


def descargar_publicos() -> tuple[pd.DataFrame, set[str]]:
    """Requiere red y `datasets` (Colab). Devuelve filas y el conjunto de claves a excluir (deepset test)."""
    from datasets import load_dataset

    filas: list[dict] = []
    excluir: set[str] = set()

    # deepset: train -> entrenamiento; test -> SOLO evaluación externa (se excluye)
    ds = load_dataset("deepset/prompt-injections")
    excluir |= {_clave(t) for t in ds["test"]["text"]}
    for i, (t, l) in enumerate(zip(ds["train"]["text"], ds["train"]["label"])):
        et = _reasignar_inyeccion(t) if l == 1 else "benigno"
        filas.append(_fila(t, et, "deepset", "en", i))

    # xTRam1/safe-guard-prompt-injection
    sg = load_dataset("xTRam1/safe-guard-prompt-injection", split="train").to_pandas()
    for etq, l in (("inj", 1), ("ben", 0)):
        sub = _muestra(sg[sg["label"] == l], 450 if l else 250)
        for i, t in enumerate(sub["text"]):
            et = _reasignar_inyeccion(t) if l == 1 else "benigno"
            filas.append(_fila(t, et, "xtram1", "en", f"{etq}{i}"))

    # jackhhao/jailbreak-classification
    jb = load_dataset("jackhhao/jailbreak-classification", split="train").to_pandas()
    for tipo, et, n in (("jailbreak", "jailbreak", 300), ("benign", "benigno", 120)):
        for i, t in enumerate(_muestra(jb[jb["type"] == tipo], n)["prompt"]):
            filas.append(_fila(t, et, "jackhhao", "en", f"{tipo}{i}"))

    # JBB-Behaviors: benignos = hard negatives; dañinos -> jailbreak (peticiones dañinas, ruido de etiqueta reconocido)
    try:
        for split, et, n in (("benign", "benigno", 100), ("harmful", "jailbreak", 60)):
            d = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=split).to_pandas()
            for i, t in enumerate(_muestra(d, n)["Goal"]):
                filas.append(_fila(t, et, f"jbb_{split}", "en", i))
    except Exception as e:  # noqa: BLE001 - corpus opcional
        print("JBB omitido:", e)

    df = pd.DataFrame(filas)
    # data_leakage adicional por regex entre benignos/inyecciones que piden credenciales
    m = (df["etiqueta"] == "prompt_injection") & df["texto"].str.contains(RE_LEAK_DATOS)
    df.loc[m, "etiqueta"] = "data_leakage"
    return df, excluir


def particionar(df: pd.DataFrame) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    _, idx_test = next(sgkf.split(df, df["etiqueta"], groups=df["plantilla_id"]))
    df = df.copy()
    df["split"] = "train"
    df.iloc[idx_test, df.columns.get_loc("split")] = "test"
    return df


def construir(con_red: bool = True) -> pd.DataFrame:
    partes = [cargar_propios()]
    excluir_extra: set[str] = set()
    if con_red:
        try:
            pub, excluir_extra = descargar_publicos()
            partes.append(pub)
        except Exception as e:  # noqa: BLE001 - sin red o sin `datasets`
            print("Corpus públicos no disponibles, solo propios:", type(e).__name__, e)
    df = pd.concat(partes, ignore_index=True)
    df = df[df["texto"].str.len() > 3]
    df["_k"] = df["texto"].map(_clave)
    df = df[~df["_k"].isin(exclusiones(excluir_extra))]
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    df = df[df["etiqueta"].isin(ETIQUETAS)].reset_index(drop=True)
    return particionar(df)


def guardar(df: pd.DataFrame) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET, index=False)
    RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    (df.groupby(["etiqueta", "fuente", "idioma"]).size().rename("filas").reset_index().to_csv(RESUMEN, index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-red", action="store_true")
    a = ap.parse_args()
    d = construir(con_red=not a.sin_red)
    guardar(d)
    print(d.groupby(["etiqueta", "split"]).size().unstack(fill_value=0))
