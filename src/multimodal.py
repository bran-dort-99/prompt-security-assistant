"""Integración multimodal: imagen -> CNN (vector) -> QR/OCR -> clasificador de texto -> categoría de ataque.

La CNN decide *qué hay* en la imagen (foto natural, texto incrustado o código QR); la *intención* la
decide el mismo clasificador de texto que usa el chatbot sobre lo extraído (texto OCR o payload del QR).
Todos los componentes son inyectables (`cnn`, `ocr`, `clasificador`) para poder probar sin modelos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

ATAQUES = {"prompt_injection", "indirect_injection", "jailbreak", "system_prompt_leakage", "data_leakage", "excessive_agency"}
UMBRAL_CNN = 0.60    # por debajo, se corre OCR/QR aunque la CNN diga "foto_natural"
UMBRAL_TEXTO = 0.45  # igual que el umbral de respaldo del chatbot (D5)
ACORTADORES = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "ow.ly", "rebrand.ly", "shorturl.at"}
RE_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

Predictor = Callable[[str], tuple[str, float, dict[str, float]]]


@dataclass
class ResultadoImagen:
    ruta: str
    clase_cnn: str
    prob_cnn: float
    probs_cnn: dict[str, float] = field(default_factory=dict)
    texto_extraido: str = ""
    payload_qr: str | None = None
    etiqueta_texto: str | None = None
    prob_texto: float = 0.0
    senales_url: list[str] = field(default_factory=list)
    categoria: str | None = None   # categoría de ataque (clave de la KB) o None
    riesgo: bool = False
    fuente_texto: str = ""         # "ocr" | "qr"
    notas: list[str] = field(default_factory=list)


def analizar_url(payload: str) -> list[str]:
    """Reglas de URL: esquema peligroso, IP literal, acortadores, HTTP sin TLS, dominio con homoglifos de dígitos."""
    senales: list[str] = []
    p = payload.strip()
    bajo = p.lower()
    if bajo.startswith(("javascript:", "data:", "vbscript:", "file:")):
        return [f"esquema peligroso ({bajo.split(':', 1)[0]}:)"]
    if not re.match(r"^[a-z][a-z0-9+.-]*://", bajo):
        if re.match(r"^(www\.)?[\w-]+(\.[\w-]+)+(/|$)", bajo):
            bajo = "http://" + bajo
        else:
            return []
    u = urlparse(bajo)
    host = (u.hostname or "")
    if RE_IP.match(host):
        senales.append("IP literal en lugar de dominio")
    if host in ACORTADORES:
        senales.append("acortador de enlaces")
    if u.scheme == "http":
        senales.append("HTTP sin cifrado")
    if re.search(r"[a-z][01][a-z]", host.split(".")[0]) and "-" in host:
        senales.append("dominio con dígitos tipo homoglifo")
    if host.endswith((".example", ".invalid", ".test", ".zip", ".mov")) or any(k in host for k in ("login-", "verify-", "secure-")):
        senales.append("dominio sospechoso")
    return senales


def decodificar_qr(ruta: str | Path) -> str | None:
    """cv2.QRCodeDetector con reintentos (escala de grises, ampliación, umbral)."""
    import cv2

    img = cv2.imread(str(ruta))
    if img is None:
        return None
    det = cv2.QRCodeDetector()
    variantes = [img]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variantes += [gris, cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                  cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)]
    if max(img.shape[:2]) > 1400:
        r = 1400 / max(img.shape[:2])
        variantes.append(cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA))
    for v in variantes:
        try:
            valor, _, _ = det.detectAndDecode(v)
        except cv2.error:
            continue
        if valor:
            return valor
    return None


_OCR = None


def ocr_easyocr(ruta: str | Path, lado_max: int = 1024) -> str:
    """EasyOCR es/en con inicialización perezosa y cacheada; reduce imágenes grandes para ir más rápido."""
    global _OCR
    import numpy as np
    from PIL import Image, ImageOps

    if _OCR is None:
        import easyocr

        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            gpu = False
        _OCR = easyocr.Reader(["es", "en"], gpu=gpu, verbose=False)
    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    if max(im.size) > lado_max:
        im.thumbnail((lado_max, lado_max))
    return " ".join(_OCR.readtext(np.array(im), detail=0, paragraph=True)).strip()


class AnalizadorImagen:
    def __init__(self, cnn: Predictor | None = None, ocr: Callable[[str], str] | None = None, clasificador: Predictor | None = None,
                 decodificador_qr: Callable[[str], str | None] | None = None, umbral_cnn: float = UMBRAL_CNN,
                 umbral_texto: float = UMBRAL_TEXTO, estricto: bool = False) -> None:
        self._cnn, self._ocr, self._clf, self._qr = cnn, ocr, clasificador, decodificador_qr
        self.umbral_cnn, self.umbral_texto, self.estricto = umbral_cnn, umbral_texto, estricto

    # componentes con carga perezosa (los modelos pesan; solo se cargan si se usan)
    def cnn(self, ruta: str):
        if self._cnn is None:
            from src.vision import modelo
            self._cnn = lambda r: modelo.predecir(r)
        return self._cnn(ruta)

    def ocr(self, ruta: str) -> str:
        return (self._ocr or ocr_easyocr)(ruta)

    def clasificador(self, texto: str):
        if self._clf is None:
            from src.nlp import clasificador_texto
            self._clf = clasificador_texto.predecir
        return self._clf(texto)

    def qr(self, ruta: str) -> str | None:
        return (self._qr or decodificar_qr)(ruta)

    def analizar(self, ruta: str | Path) -> ResultadoImagen:
        ruta = str(ruta)
        if not Path(ruta).exists():
            raise FileNotFoundError(ruta)
        clase, prob, probs = self.cnn(ruta)
        r = ResultadoImagen(ruta, clase, prob, probs)
        dudosa = prob < self.umbral_cnn
        if dudosa:
            r.notas.append(f"CNN con baja confianza ({prob:.2f} < {self.umbral_cnn}): se corren QR y OCR")
        if self.estricto:
            r.notas.append("modo estricto: QR y OCR siempre")

        textos: list[tuple[str, str]] = []  # (fuente, texto)
        if clase == "codigo_qr" or dudosa or self.estricto:
            payload = self.qr(ruta)
            if payload:
                r.payload_qr, r.fuente_texto = payload, "qr"
                r.senales_url = analizar_url(payload)
                textos.append(("qr", payload))
            elif clase == "codigo_qr":
                r.notas.append("la CNN detecta un QR pero no se pudo decodificar")
        if clase == "texto_incrustado" or dudosa or self.estricto:
            texto = self.ocr(ruta)
            r.texto_extraido = texto
            if texto.strip():
                textos.append(("ocr", texto))
            else:
                r.notas.append("OCR sin texto legible (posible bajo contraste u ofuscación)")

        # decisión: se queda el texto con mayor probabilidad de ataque
        mejor: tuple[str, str, float] | None = None
        for fuente, t in textos:
            et, p, probs_t = self.clasificador(t)
            if et in ATAQUES and p >= self.umbral_texto and (mejor is None or p > mejor[2]):
                mejor = (fuente, et, p)
            if r.etiqueta_texto is None or (et in ATAQUES and p >= r.prob_texto):
                r.etiqueta_texto, r.prob_texto, r.fuente_texto = et, p, fuente
        if mejor:
            r.fuente_texto, r.etiqueta_texto, r.prob_texto = mejor
            # el vector es la imagen: una inyección dentro de una imagen es indirecta (AML.T0051.001 / LLM01)
            r.categoria = "indirect_injection" if mejor[1] == "prompt_injection" else mejor[1]
            r.riesgo = True
        elif r.senales_url:
            r.categoria, r.riesgo = "indirect_injection", True
            r.notas.append("QR con enlace sospechoso: " + ", ".join(r.senales_url))
        return r
