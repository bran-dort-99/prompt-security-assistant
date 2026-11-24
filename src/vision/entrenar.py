"""Entrenamiento en 2 etapas (backbone congelado -> fine-tune de los últimos 2 bloques) + evaluación.

    python -m src.vision.entrenar --datos data/imagenes --reales data/imagenes_reales
Funciona en CUDA (WSL2/Colab), MPS (Mac) o CPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from src.vision import modelo as M

RAIZ = Path(__file__).resolve().parent.parent.parent
EVID = RAIZ / "evidencias" / "final"


def fijar_semillas(s: int = 42) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def cargadores(datos: Path, batch: int = 32, workers: int = 4):
    ds = {"train": datasets.ImageFolder(datos / "train", M.transform_train()),
          "val": datasets.ImageFolder(datos / "val", M.transform_eval()),
          "test_sintetico": datasets.ImageFolder(datos / "test_sintetico", M.transform_eval())}
    assert ds["train"].classes == sorted(M.CLASES), ds["train"].classes
    dl = {k: DataLoader(v, batch_size=batch, shuffle=(k == "train"), num_workers=workers, pin_memory=torch.cuda.is_available()) for k, v in ds.items()}
    return ds, dl


def _epoca(m, dl, dev, opt=None, crit=None):
    train = opt is not None
    m.train(train)
    perdida, aciertos, n, ys, ps = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            out = m(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            perdida += loss.item() * len(y); n += len(y)
            p = out.argmax(1); aciertos += (p == y).sum().item()
            ys += y.cpu().tolist(); ps += p.cpu().tolist()
    return perdida / n, aciertos / n, f1_score(ys, ps, average="macro", zero_division=0)


def entrenar(datos: Path, backbone="efficientnet_b0", ep1=3, ep2=5, batch=32, workers=4, weights="imagenet", salida: Path = M.RUTA_MODELO, semilla=42, paciencia=3):
    fijar_semillas(semilla)
    dev = M.dispositivo()
    ds, dl = cargadores(datos, batch, workers)
    m = M.construir_modelo(backbone, len(ds["train"].classes), weights).to(dev)
    conteo = np.bincount(ds["train"].targets)
    pesos = torch.tensor(conteo.sum() / (len(conteo) * conteo), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=pesos)
    hist = []
    mejor, mejor_f1 = copy.deepcopy(m.state_dict()), -1.0

    def correr(etapa, epocas, lr):
        nonlocal mejor, mejor_f1
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
        sin_mejora = 0
        for e in range(epocas):
            t = time.time()
            tl, ta, tf = _epoca(m, dl["train"], dev, opt, crit)
            vl, va, vf = _epoca(m, dl["val"], dev, None, crit)
            hist.append({"etapa": etapa, "epoca": e + 1, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va, "val_f1": vf, "seg": round(time.time() - t, 1)})
            print(hist[-1])
            if vf > mejor_f1:
                mejor_f1, mejor, sin_mejora = vf, copy.deepcopy(m.state_dict()), 0
            else:
                sin_mejora += 1
                if etapa == 2 and sin_mejora >= paciencia:
                    break

    M.congelar_backbone(m); correr(1, ep1, 1e-3)
    M.descongelar_ultimos(m, 2); correr(2, ep2, 1e-4)
    m.load_state_dict(mejor)
    M.guardar(m, salida, backbone, ds["train"].classes)
    return m, ds, hist


def evaluar_dl(m, dl, clases, dev):
    m.eval(); ys, ps, rutas = [], [], []
    with torch.no_grad():
        for x, y in dl:
            ps += m(x.to(dev)).softmax(1).cpu().argmax(1).tolist(); ys += y.tolist()
    return ys, ps


def metricas(ys, ps, clases):
    rep = classification_report(ys, ps, labels=list(range(len(clases))), target_names=clases, output_dict=True, zero_division=0)
    return {"accuracy": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"], "por_clase": {c: rep[c] for c in clases}, "matriz": confusion_matrix(ys, ps, labels=range(len(clases))).tolist()}


def figura_matriz(mat, clases, titulo, ruta: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = np.array(mat); fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(mat, cmap="Blues"); ax.set_xticks(range(len(clases)), clases, rotation=30, ha="right"); ax.set_yticks(range(len(clases)), clases)
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, mat[i, j], ha="center", va="center", color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title(titulo); fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)


def figura_curvas(hist, ruta: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = range(1, len(hist) + 1); fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(x, [h["train_loss"] for h in hist], label="train"); ax[0].plot(x, [h["val_loss"] for h in hist], label="val"); ax[0].set_title("Pérdida"); ax[0].legend()
    ax[1].plot(x, [h["train_acc"] for h in hist], label="train"); ax[1].plot(x, [h["val_acc"] for h in hist], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    for a in ax:
        a.set_xlabel("época (acumulada)")
        a.axvline(sum(1 for h in hist if h["etapa"] == 1) + .5, color="gray", ls="--")
    fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)


def evaluar_reales(m, reales: Path, clases, dev):
    """Test real: `manifest.csv` con columnas archivo,clase. Devuelve métricas y la lista de errores."""
    import csv
    filas = list(csv.DictReader((reales / "manifest.csv").open(encoding="utf-8")))
    ys, ps, errores = [], [], []
    for r in filas:
        if r["clase"] not in clases or not (reales / r["archivo"]).exists():
            continue
        c, p, _ = M.predecir(reales / r["archivo"], m, clases)
        ys.append(clases.index(r["clase"])); ps.append(clases.index(c))
        if c != r["clase"]:
            errores.append({"archivo": r["archivo"], "real": r["clase"], "predicha": c, "prob": round(p, 3)})
    return (metricas(ys, ps, clases) if ys else None), errores, len(ys)


def principal():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(RAIZ / "data" / "imagenes")); ap.add_argument("--reales", default=str(RAIZ / "data" / "imagenes_reales"))
    ap.add_argument("--backbone", default="efficientnet_b0"); ap.add_argument("--ep1", type=int, default=3); ap.add_argument("--ep2", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--evid", default=str(EVID)); ap.add_argument("--solo-reales", action="store_true", help="no reentrena: evalúa el modelo guardado en las fotos reales y actualiza cnn_metricas.json")
    a = ap.parse_args()
    evid = Path(a.evid); evid.mkdir(parents=True, exist_ok=True)
    if a.solo_reales:
        m, clases = M.cargar(M.RUTA_MODELO, M.dispositivo()); dev = M.dispositivo()
        ruta = evid / "cnn_metricas.json"; out = json.loads(ruta.read_text(encoding="utf-8"))
        mr, errores, n = evaluar_reales(m, Path(a.reales), clases, dev)
        out["test_real"] = {"n": n, **mr, "errores": errores}
        figura_matriz(mr["matriz"], clases, f"CNN · test real (n={n})", evid / "cnn_matriz_real.png")
        ruta.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("accuracy real:", round(mr["accuracy"], 3), "n =", n); print("errores:", errores); return
    t0 = time.time()
    m, ds, hist = entrenar(Path(a.datos), a.backbone, a.ep1, a.ep2, a.batch, a.workers)
    dev = M.dispositivo(); clases = ds["train"].classes
    ys, ps = evaluar_dl(m, DataLoader(ds["test_sintetico"], batch_size=64, num_workers=a.workers), clases, dev)
    out = {"dispositivo": str(dev), "backbone": a.backbone, "segundos_totales": round(time.time() - t0), "historial": hist, "test_sintetico": metricas(ys, ps, clases)}
    figura_matriz(out["test_sintetico"]["matriz"], clases, "CNN · test sintético", evid / "cnn_matriz_sintetico.png")
    figura_curvas(hist, evid / "cnn_curvas.png")
    reales = Path(a.reales)
    if (reales / "manifest.csv").exists():
        mr, errores, n = evaluar_reales(m, reales, clases, dev)
        if mr:
            out["test_real"] = {"n": n, **mr, "errores": errores}
            figura_matriz(mr["matriz"], clases, f"CNN · test real (n={n})", evid / "cnn_matriz_real.png")
    (evid / "cnn_metricas.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("accuracy sintético:", round(out["test_sintetico"]["accuracy"], 3), "| F1 macro:", round(out["test_sintetico"]["f1_macro"], 3))
    if "test_real" in out:
        print("accuracy real:", round(out["test_real"]["accuracy"], 3), "n =", out["test_real"]["n"])


if __name__ == "__main__":
    principal()
