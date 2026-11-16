"""Genera los diagramas del informe (docs/avance2/figuras/). Uso: python scripts/diagramas.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SALIDA = Path(__file__).resolve().parent.parent / "docs" / "avance2" / "figuras"
AZUL, GRIS, NARANJA, VERDE, LILA, BLANCO = "#12355b", "#c9d2dc", "#f4b183", "#a9d18e", "#d5c8ef", "#ffffff"


def caja(ax, x, y, w, h, texto, color, tam=8.5, negrita=False, tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=AZUL, lw=1.2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=tam, color=tc, fontweight="bold" if negrita else "normal", linespacing=1.25)


def flecha(ax, p, q, texto=None, tam=7.5, desp=(0, 0.12), conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=AZUL, connectionstyle=conn))
    if texto:
        ax.text((p[0] + q[0]) / 2 + desp[0], (p[1] + q[1]) / 2 + desp[1], texto, ha="center", va="bottom", fontsize=tam, color="#333333", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))


def cnn():
    fig, ax = plt.subplots(figsize=(11, 3.9)); ax.set_xlim(0, 22); ax.set_ylim(0, 7.4); ax.axis("off")
    y, h = 3.0, 2.1
    caja(ax, 0.2, y, 2.3, h, "Imagen RGB\n224 × 224 × 3\n(normalizada\ncon ImageNet)", BLANCO)
    caja(ax, 3.2, y, 3.6, h, "Stem + MBConv\nbloques 1–6\n(preentrenados)", GRIS)
    caja(ax, 7.5, y, 2.4, h, "MBConv 7", NARANJA)
    caja(ax, 10.6, y, 2.5, h, "Conv 1×1\n→ 1280 mapas", NARANJA)
    caja(ax, 13.8, y, 2.0, h, "Pool global\npromedio\n→ vector 1280", BLANCO, tam=8)
    caja(ax, 16.5, y, 1.9, h, "Dropout\n0.3", VERDE)
    caja(ax, 19.0, y, 2.8, h, "Linear\n1280 → 3\n+ Softmax", VERDE)
    for a, b in [(2.5, 3.2), (6.8, 7.5), (9.9, 10.6), (13.1, 13.8), (15.8, 16.5), (18.4, 19.0)]:
        flecha(ax, (a, y + h / 2), (b, y + h / 2))
    ax.text(21.4, y - 0.35, "codigo_qr · foto_natural\ntexto_incrustado", ha="right", va="top", fontsize=8, color=AZUL, fontweight="bold")
    # llaves de etapas
    ax.plot([3.2, 13.1], [6.0, 6.0], color=AZUL, lw=1); ax.text(8.15, 6.15, "Backbone: EfficientNet-B0 preentrenada en ImageNet  (features[0–8], ~4.0 M parámetros)", ha="center", fontsize=8.5, color=AZUL, fontweight="bold")
    ax.plot([16.5, 21.8], [6.0, 6.0], color=AZUL, lw=1); ax.text(19.15, 6.15, "Cabeza nueva (3,843 parámetros)", ha="center", fontsize=8.5, color=AZUL, fontweight="bold")
    # leyenda
    for i, (c, t) in enumerate([(GRIS, "Congelado en ambas etapas"), (NARANJA, "Se entrena solo en la etapa 2 (1.13 M parámetros, lr 1e-4)"), (VERDE, "Se entrena en las etapas 1 y 2 (lr 1e-3 → 1e-4)")]):
        ax.add_patch(FancyBboxPatch((0.4 + i * 7.4, 0.35), 0.5, 0.5, boxstyle="round,pad=0.01", fc=c, ec=AZUL, lw=1))
        ax.text(1.1 + i * 7.4, 0.6, t, va="center", fontsize=8)
    fig.tight_layout(pad=0.4); fig.savefig(SALIDA / "arquitectura_cnn.png", dpi=200); plt.close(fig)


def cascada():
    fig, ax = plt.subplots(figsize=(11, 6.4)); ax.set_xlim(0, 22); ax.set_ylim(0, 13); ax.axis("off")
    caja(ax, 0.2, 5.6, 2.6, 1.8, "Imagen\n(foto, captura,\nQR)", BLANCO)
    caja(ax, 3.6, 5.6, 3.0, 1.8, "CNN\nEfficientNet-B0\n3 clases", NARANJA, negrita=True)
    flecha(ax, (2.8, 6.5), (3.6, 6.5))
    caja(ax, 8.3, 10.3, 3.6, 1.7, "Decodificador de QR\n(OpenCV)\n+ reglas de URL", LILA)
    caja(ax, 8.3, 5.65, 3.6, 1.7, "OCR\n(EasyOCR es/en)", LILA)
    caja(ax, 8.3, 1.0, 3.6, 1.7, "Sin análisis adicional\n(modo --estricto: OCR)", GRIS, tam=8)
    flecha(ax, (6.6, 7.1), (8.3, 11.0), "codigo_qr", desp=(-0.9, 0.25))
    flecha(ax, (6.6, 6.5), (8.3, 6.5), "texto\nincrustado", desp=(0, 0.12), tam=7)
    flecha(ax, (6.6, 5.9), (8.3, 2.0), "foto_natural\n(confianza ≥ 0.60)", desp=(-1.0, -0.6))
    ax.text(5.1, 4.45, "confianza < 0.60 →\nQR y OCR a la vez", ha="center", fontsize=7.5, color=AZUL, style="italic")
    caja(ax, 13.6, 5.2, 3.5, 2.6, "Clasificador de intención\nTF-IDF + SVM calibrado\n+ reglas de respaldo\n(umbral 0.30)", VERDE, negrita=True)
    flecha(ax, (11.9, 11.1), (14.9, 7.8), "payload", desp=(0.4, 0.25))
    flecha(ax, (11.9, 6.5), (13.6, 6.5), "texto", desp=(0, 0.1))
    flecha(ax, (11.9, 1.85), (18.3, 2.3), "sin hallazgos", desp=(0, 0.08), tam=7.5)
    caja(ax, 18.3, 8.0, 3.5, 2.0, "Categoría de ataque\n(vector = imagen;\nprompt_injection →\nindirect_injection)", BLANCO, tam=8)
    caja(ax, 18.3, 4.8, 3.5, 2.2, "Mapeo a controles\nOWASP LLM Top 10\nMITRE ATLAS\nNIST AI RMF", BLANCO, tam=8.5)
    caja(ax, 18.3, 1.0, 3.5, 2.7, "Respuesta del asistente\n+ «Evidencia visual»:\nclase de la CNN y prob.,\ntexto del OCR, payload\ndel QR, señales de URL", BLANCO, tam=8)
    flecha(ax, (17.1, 6.9), (18.3, 9.0)); flecha(ax, (17.1, 6.2), (18.3, 5.9)); flecha(ax, (20.05, 4.8), (20.05, 3.7))
    flecha(ax, (20.05, 8.0), (20.05, 7.0))
    ax.text(11, 0.2, "Texto escrito por el usuario: mismo clasificador y mismo mapeo, sin pasar por la CNN.", ha="center", fontsize=8.5, color=AZUL, style="italic")
    fig.tight_layout(pad=0.4); fig.savefig(SALIDA / "cascada_multimodal.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True); cnn(); cascada(); print("OK:", SALIDA)
