"""Clasificador de intenciones (12 etiquetas, D4) y experimentos (D7, D8).

    python -m src.nlp.clasificador_texto            # corre todo y escribe evidencias + modelo
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.nlp.preprocesamiento import normalizar_texto, preprocesar

RAIZ = Path(__file__).resolve().parent.parent.parent
MODELO = RAIZ / "models" / "clasificador_intenciones.joblib"
EVID = RAIZ / "evidencias" / "avance2"
ATAQUES = {"prompt_injection", "indirect_injection", "jailbreak", "system_prompt_leakage", "data_leakage", "excessive_agency"}
SEMILLA = 42

REPRESENTACIONES = ("bow", "tfidf", "tfidf+chars")
CLASIFICADORES = ("cnb", "svc")


def _analizador(texto: str, **kw) -> list[str]:
    return preprocesar(texto, **kw)


def construir_pipeline(representacion: str = "tfidf+chars", clasificador: str = "svc", *,
                       stopwords: bool = True, stemming: bool = True, lista_blanca: bool = True) -> Pipeline:
    analizador = partial(_analizador, stopwords=stopwords, stemming=stemming, lista_blanca=lista_blanca)
    if representacion == "bow":
        vec = CountVectorizer(analyzer=analizador, ngram_range=(1, 1))
    elif representacion == "tfidf":
        vec = TfidfVectorizer(tokenizer=analizador, preprocessor=None, lowercase=False, token_pattern=None,
                              ngram_range=(1, 2), sublinear_tf=True)
    elif representacion == "tfidf+chars":
        vec = FeatureUnion([
            ("palabras", TfidfVectorizer(tokenizer=analizador, preprocessor=None, lowercase=False, token_pattern=None,
                                         ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalizar_texto,
                                      sublinear_tf=True, min_df=2)),
        ])
    else:
        raise ValueError(representacion)
    if clasificador == "cnb":
        clf = ComplementNB()
    elif clasificador == "svc":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    else:
        raise ValueError(clasificador)
    return Pipeline([("vec", vec), ("clf", clf)])


def entrenar(df: pd.DataFrame, **kw) -> Pipeline:
    modelo = construir_pipeline(**kw)
    modelo.fit(df["texto"], df["etiqueta"])
    return modelo


def evaluar(modelo: Pipeline, df: pd.DataFrame) -> dict:
    pred = modelo.predict(df["texto"])
    etiquetas = sorted(set(df["etiqueta"]) | set(pred))
    return {
        "accuracy": float((pred == df["etiqueta"]).mean()),
        "f1_macro": float(f1_score(df["etiqueta"], pred, average="macro", zero_division=0)),
        "f1_ponderado": float(f1_score(df["etiqueta"], pred, average="weighted", zero_division=0)),
        "reporte": classification_report(df["etiqueta"], pred, output_dict=True, zero_division=0),
        "etiquetas": etiquetas,
        "matriz": confusion_matrix(df["etiqueta"], pred, labels=etiquetas).tolist(),
    }


def guardar(modelo: Pipeline, ruta: Path = MODELO) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta, compress=3)


_CACHE: dict[str, Pipeline] = {}


def cargar(ruta: Path = MODELO) -> Pipeline:
    clave = str(ruta)
    if clave not in _CACHE:
        _CACHE[clave] = joblib.load(ruta)
    return _CACHE[clave]


def predecir(texto: str, modelo: Pipeline | None = None) -> tuple[str, float, dict[str, float]]:
    """(etiqueta, probabilidad, todas las probabilidades)."""
    modelo = modelo or cargar()
    p = modelo.predict_proba([texto])[0]
    probs = {c: float(v) for c, v in zip(modelo.classes_, p)}
    mejor = max(probs, key=probs.get)
    return mejor, probs[mejor], probs


# ---------------------------------------------------------------- experimentos
def matriz_experimentos(train: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    # Agrupada por plantilla: sin esto la CV filtra plantillas entre folds e infla el F1 (0.96 vs ~0.56 en test).
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEMILLA)
    filas = []
    for rep in REPRESENTACIONES:
        for clf in CLASIFICADORES:
            puntajes = cross_val_score(construir_pipeline(rep, clf), train["texto"], train["etiqueta"],
                                       cv=cv, groups=train["plantilla_id"], scoring="f1_macro", n_jobs=-1)
            filas.append({"representacion": rep, "clasificador": clf, "f1_macro_media": puntajes.mean(), "f1_macro_std": puntajes.std()})
    return pd.DataFrame(filas).sort_values("f1_macro_media", ascending=False).reset_index(drop=True)


def ablacion(train: pd.DataFrame, test: pd.DataFrame, rep: str, clf: str) -> pd.DataFrame:
    config = {
        "completo": {},
        "sin_stopwords": {"stopwords": False},
        "sin_stemming": {"stemming": False},
        "sin_lista_blanca": {"lista_blanca": False},
    }
    filas = []
    for nombre, kw in config.items():
        m = entrenar(train, representacion=rep, clasificador=clf, **kw)
        e = evaluar(m, test)
        filas.append({"variante": nombre, "f1_macro": e["f1_macro"], "accuracy": e["accuracy"]})
    return pd.DataFrame(filas)


def externo_binario(modelo: Pipeline, textos: list[str], etiquetas_bin: list[int]) -> dict:
    """Colapsa a ataque(1)/benigno(0). Conversacionales y `consulta_seguridad` cuentan como benignos."""
    pred = [int(predecir(t, modelo)[0] in ATAQUES) for t in textos]
    p, r, f, _ = precision_recall_fscore_support(etiquetas_bin, pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "n": len(textos)}


def cargar_externo_deepset() -> tuple[list[str], list[int]] | None:
    try:
        from datasets import load_dataset

        d = load_dataset("deepset/prompt-injections", split="test")
        return list(d["text"]), list(d["label"])
    except Exception as e:  # noqa: BLE001
        print("Test externo (deepset) no disponible:", type(e).__name__)
        return None


def bateria_ml(modelo: Pipeline, umbral: float = 0.0) -> pd.DataFrame:
    """Evalúa la batería de 36 con el ML puro (sin reglas). Ataque correcto = cualquier etiqueta de ataque."""
    bateria = json.loads((RAIZ / "data" / "bateria_prompts.json").read_text(encoding="utf-8"))
    filas = []
    for it in bateria:
        et, p, _ = predecir(it["prompt"], modelo)
        es_ataque = et in ATAQUES
        esperado_ataque = it["tipo"] != "benigno"
        filas.append({"tipo": it["tipo"], "categoria_esperada": it.get("categoria"), "prompt": it["prompt"],
                      "prediccion": et, "prob": round(p, 3), "ok_binario": es_ataque == esperado_ataque,
                      "ok_categoria": et == it.get("categoria")})
    return pd.DataFrame(filas)


def figura_matriz(ev: dict, ruta: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(ev["matriz"])
    et = ev["etiquetas"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(et)), et, rotation=60, ha="right")
    ax.set_yticks(range(len(et)), et)
    for i in range(len(et)):
        for j in range(len(et)):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión — intenciones (test)")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def correr_todo(dataset: Path | None = None, evid: Path = EVID, modelo_salida: Path = MODELO) -> dict:
    df = pd.read_csv(dataset or RAIZ / "data" / "intenciones" / "dataset.csv")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    evid.mkdir(parents=True, exist_ok=True)

    matriz = matriz_experimentos(train)
    matriz.to_csv(evid / "texto_matriz_experimentos.csv", index=False)
    mejor = matriz.iloc[0]
    rep, clf = mejor["representacion"], mejor["clasificador"]
    # probabilidades: ComplementNB también las expone (no calibradas); se elige por F1 como pide D7
    abl = ablacion(train, test, rep, clf)
    abl.to_csv(evid / "texto_ablacion.csv", index=False)

    modelo = entrenar(train, representacion=rep, clasificador=clf)
    ev = evaluar(modelo, test)
    figura_matriz(ev, evid / "texto_matriz_confusion.png")
    salida = {"mejor_config": {"representacion": rep, "clasificador": clf}, "test": {k: ev[k] for k in ("accuracy", "f1_macro", "f1_ponderado", "reporte")},
              "n_train": len(train), "n_test": len(test)}

    ext = cargar_externo_deepset()
    if ext:
        salida["externo_deepset"] = externo_binario(modelo, *ext)

    bat = bateria_ml(modelo)
    bat.to_csv(evid / "texto_bateria_ml.csv", index=False)
    salida["bateria_ml"] = {t: {"ok_binario": int(g["ok_binario"].sum()), "n": len(g)} for t, g in bat.groupby("tipo")}

    # modelo final: entrenar con TODO el dataset (train+test) para el chatbot; las métricas son del split
    guardar(entrenar(df, representacion=rep, clasificador=clf), modelo_salida)
    (evid / "texto_metricas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    return salida


if __name__ == "__main__":
    r = correr_todo()
    print(json.dumps({k: v for k, v in r.items() if k != "test"}, indent=2, ensure_ascii=False))
    print("F1 macro test:", round(r["test"]["f1_macro"], 3))
