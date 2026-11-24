"""Evaluación del clasificador de intenciones para el proyecto final (feedback del Avance 2).

Compara, entrenando ambos en el MISMO split de entrenamiento:
  - base    : datos del Avance 2 (sin las plantillas nuevas `-fNN`)
  - final   : + ejemplos en español y negativos difíciles (frontera)
y cada uno con y sin el paso de desofuscación. Conjuntos de evaluación:
  1. test interno (split agrupado por plantilla)
  2. prueba de frontera escrita a mano (data/intenciones/frontera_es_en.csv, nunca se entrena)
  3. deepset test (externo) y deepset test ofuscado con 4 técnicas (leetspeak, espaciado, Base64, homoglifos)

    python -m src.nlp.evaluar_final
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.nlp.clasificador_texto import ATAQUES, EVID, RAIZ, cargar_externo_deepset, entrenar
from src.nlp.ofuscacion import TRANSFORMACIONES, desofuscar

FRONTERA = RAIZ / "data" / "intenciones" / "frontera_es_en.csv"
TRES = ["prompt_injection", "jailbreak", "system_prompt_leakage"]
UMBRAL = 0.30


def _pred(modelo, textos, des: bool):
    textos = [desofuscar(t).texto if des else t for t in textos]
    P = modelo.predict_proba(textos)
    cl = list(modelo.classes_)
    return [cl[i] for i in P.argmax(1)], P.max(1), P, cl


def _binario(modelo, textos, y, des):
    """Detección ataque/no ataque como en el chatbot: etiqueta de ataque con prob ≥ umbral."""
    et, p, _, _ = _pred(modelo, textos, des)
    yp = [int(e in ATAQUES and q >= UMBRAL) for e, q in zip(et, p)]
    tp = sum(a and b for a, b in zip(y, yp)); pos = sum(y)
    return {"recall": tp / pos if pos else None, "precision": tp / max(sum(yp), 1), "n": len(y), "positivos": pos}


def main(evid: Path = EVID) -> dict:
    evid.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAIZ / "data" / "intenciones" / "dataset.csv")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    nuevo = train["plantilla_id"].astype(str).str.contains(r"-f\d\d$")
    met = evid / "texto_metricas.json"  # misma configuración que eligió la CV de clasificador_texto
    cfg = json.loads(met.read_text(encoding="utf-8"))["mejor_config"] if met.exists() else {"representacion": "tfidf", "clasificador": "svc"}
    modelos = {"base": entrenar(train[~nuevo], **cfg), "final": entrenar(train, **cfg)}
    fr = pd.read_csv(FRONTERA)
    ext = cargar_externo_deepset()
    out: dict = {"config": cfg, "n_train_base": int((~nuevo).sum()), "n_train_final": len(train), "n_test": len(test), "n_frontera": len(fr), "resultados": {}}
    filas = []
    for nombre, m in modelos.items():
        for des in (False, True):
            clave = f"{nombre}{'+desofuscar' if des else ''}"
            r = {}
            et, *_ = _pred(m, test["texto"].tolist(), des)
            r["test_f1_macro"] = f1_score(test["etiqueta"], et, average="macro")
            rep = classification_report(test["etiqueta"], et, output_dict=True, zero_division=0)
            r["test_f1_tres"] = {c: rep[c]["f1-score"] for c in TRES}
            ef, *_ = _pred(m, fr["texto"].tolist(), des)
            r["frontera_accuracy"] = float((pd.Series(ef) == fr["etiqueta"].values).mean())
            mask = fr["etiqueta"].isin(TRES).values
            r["frontera_accuracy_tres"] = float((pd.Series(ef)[mask].values == fr["etiqueta"].values[mask]).mean())
            r["frontera_falsos_positivos_benignos"] = int(sum(e in ATAQUES for e, y in zip(ef, fr["etiqueta"]) if y in ("benigno", "consulta_seguridad")))
            if ext:
                X, y = ext
                r["deepset"] = _binario(m, X, y, des)
                ataques = [t for t, l in zip(X, y) if l == 1]
                r["deepset_ofuscado"] = {n: _binario(m, [f(t) for t in ataques], [1] * len(ataques), des)["recall"] for n, f in TRANSFORMACIONES.items()}
            out["resultados"][clave] = r
            fila = {"modelo": clave, "test_f1_macro": r["test_f1_macro"], **{f"test_f1_{c}": v for c, v in r["test_f1_tres"].items()},
                    "frontera_acc": r["frontera_accuracy"], "frontera_acc_3clases": r["frontera_accuracy_tres"], "frontera_fp_benignos": r["frontera_falsos_positivos_benignos"]}
            if ext:
                fila["deepset_recall"] = r["deepset"]["recall"]; fila["deepset_precision"] = r["deepset"]["precision"]
                fila.update({f"ofuscado_{n}_recall": v for n, v in r["deepset_ofuscado"].items()})
            filas.append(fila)
            if nombre == "final" and des:  # matriz 3×3 de la frontera en el test interno, modelo final
                sub = test["etiqueta"].isin(TRES).values
                cm = confusion_matrix(test["etiqueta"].values[sub], pd.Series(et).values[sub], labels=TRES + ["benigno"])
                out["matriz_tres_test_final"] = cm[:3].tolist()
                errores = [{"texto": t, "real": y, "predicha": e} for t, y, e in zip(fr["texto"], fr["etiqueta"], ef) if y != e]
                out["frontera_errores_final"] = errores
    out["hibrido_frontera"] = hibrido_frontera(fr)
    pd.DataFrame(filas).round(3).to_csv(evid / "nlp_comparacion.csv", index=False)
    (evid / "nlp_comparacion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _figura(out, evid / "nlp_frontera_matriz.png")
    return out


def hibrido_frontera(fr: pd.DataFrame) -> dict:
    """Chatbot híbrido completo (reglas + ML + desofuscación) sobre la prueba de frontera: modelo del Avance 2 vs. final.
    Justo: la prueba de frontera nunca entró al entrenamiento de ninguno de los dos modelos."""
    from src.chatbot import AsistenteSeguridad
    from src.nlp.clasificador_texto import MODELO
    res = {}
    for nombre, ruta in (("avance2", RAIZ / "models" / "clasificador_intenciones_avance2.joblib"), ("final", MODELO)):
        if not ruta.exists():
            continue
        from src.nlp.clasificador_texto import cargar
        bot = AsistenteSeguridad(modelo_texto=cargar(ruta))
        pred = [bot._clasificar(t)[0] for t in fr["texto"]]
        ataque_real = fr["etiqueta"].isin(ATAQUES).values
        ok_cat = [p == y for p, y in zip(pred, fr["etiqueta"]) if y in ATAQUES]
        ok_bin = [(p is not None) == a for p, a in zip(pred, ataque_real)]
        res[nombre] = {"categoria_exacta_ataques": float(sum(ok_cat) / len(ok_cat)), "binario": float(sum(ok_bin) / len(ok_bin)),
                       "falsos_positivos": int(sum(p is not None and not a for p, a in zip(pred, ataque_real)))}
    return res


def _figura(out, ruta):
    cm = out["matriz_tres_test_final"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    cols = TRES + ["benigno"]
    ax.set_xticks(range(4), [c.replace("_", "\n") for c in cols], fontsize=7); ax.set_yticks(range(3), [c.replace("_", "\n") for c in TRES], fontsize=7)
    for i, fila in enumerate(cm):
        for j, v in enumerate(fila):
            ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > max(map(max, cm)) / 2 else "black")
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title("Frontera entre categorías (test interno, modelo final)", fontsize=9)
    fig.tight_layout(); fig.savefig(ruta, dpi=160); plt.close(fig)


if __name__ == "__main__":
    from src.nlp.evaluar_final import main as _m
    r = _m()
    print(pd.read_csv(EVID / "nlp_comparacion.csv").to_string(index=False))
    print("híbrido en frontera:", r["hibrido_frontera"])
