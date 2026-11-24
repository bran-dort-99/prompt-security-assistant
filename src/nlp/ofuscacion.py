"""Detección y reversión de ofuscación antes de clasificar (proyecto final, feedback del Avance 2).

Técnicas cubiertas (cada una se reporta por nombre en `tecnicas`):
- `ancho_cero`  : caracteres invisibles (U+200B…U+200D, U+2060, U+FEFF, guion suave).
- `homoglifos`  : letras cirílicas/griegas que imitan latinas (а→a, о→o, р→p…) y formas de ancho completo (NFKC).
- `espaciado`   : letras separadas («i g n o r a», «i.g.n.o.r.a»).
- `leetspeak`   : sustitución de letras por dígitos/símbolos («1gn0r4» → «ignora»).
- `base64`, `hex`, `url`: fragmentos codificados que decodifican a texto legible; se sustituyen por el texto decodificado.

El resultado se usa como entrada del clasificador y de las reglas. El texto original no se pierde: la respuesta
del asistente muestra qué técnicas se detectaron y el contenido decodificado.

    >>> desofuscar("1gn0r4 l4s 1nstrucc10nes").texto
    'ignora las instrucciones'
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import unquote

ANCHO_CERO = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
HOMOGLIFOS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
    "α": "a", "ο": "o", "ρ": "p", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "υ": "u", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
})
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "|": "l", "€": "e"}
LEET_CHARS = re.escape("".join(LEET))

RE_ESPACIADO = re.compile(r"(?<![\w])((?:\w[ .\-_*·]){2,}\w)(?![\w])")
RE_TOKEN_LEET = re.compile(rf"[\w{LEET_CHARS}]+")
RE_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/_-]{12,}={0,2})(?![A-Za-z0-9+/=_-])")
RE_HEX = re.compile(r"(?<![0-9A-Fa-f])((?:[0-9A-Fa-f]{2}){8,})(?![0-9A-Fa-f])")
RE_URLENC = re.compile(r"(?:%[0-9A-Fa-f]{2}){3,}")


@dataclass
class ResultadoOfuscacion:
    original: str
    texto: str
    tecnicas: list[str] = field(default_factory=list)
    decodificado: list[str] = field(default_factory=list)

    @property
    def ofuscado(self) -> bool:
        return bool(self.tecnicas)


def _legible(t: str, min_len: int = 8) -> bool:
    """Texto decodificado plausible: imprimible, con letras y (casi siempre) espacios."""
    if len(t) < min_len or "�" in t:
        return False
    imprimibles = sum(c.isprintable() or c in "\n\t" for c in t) / len(t)
    letras = sum(c.isalpha() for c in t) / len(t)
    return imprimibles >= 0.95 and letras >= 0.55 and (" " in t or len(t) <= 24)


def _b64(token: str) -> str | None:
    s = token.rstrip("=")
    if len(s) < 12 or not (re.search(r"[A-Z]", s) and re.search(r"[a-z]", s)):
        return None
    relleno = s + "=" * (-len(s) % 4)
    for dec in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            t = dec(relleno).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if _legible(t):
            return t
    return None


def _hex(token: str) -> str | None:
    try:
        t = bytes.fromhex(token).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return t if _legible(t) else None


def _leet_token(tok: str) -> str | None:
    """Convierte un token si mezcla letras con sustitutos leet (evita números puros, IDs como LLM01 y años)."""
    if not re.search(r"[A-Za-zÀ-ÿ]", tok) or len(tok) < 2:
        return None
    n_leet = sum(c in LEET for c in tok)
    if n_leet == 0 or n_leet / len(tok) < 0.2:
        return None
    if re.fullmatch(r"[A-Za-z]{2,}\d{1,3}(\.\d+)*", tok) or re.fullmatch(r"\d+(k|g|d|p|gb|mb|tb|fa|x|h|m|s|am|pm)", tok, re.I):
        return None  # LLM01, mp3, gpt4: sufijos numéricos legítimos
    return "".join(LEET.get(c, c) for c in tok).lower()


def desofuscar(texto: str) -> ResultadoOfuscacion:
    r = ResultadoOfuscacion(original=texto, texto=texto)
    t = texto

    t2 = t.translate(ANCHO_CERO)
    if t2 != t:
        r.tecnicas.append("ancho_cero"); t = t2
    t2 = unicodedata.normalize("NFKC", t).translate(HOMOGLIFOS)
    if t2 != t:
        r.tecnicas.append("homoglifos"); t = t2

    # Codificaciones (antes del leetspeak: Base64 contiene dígitos)
    def _sub(regex, fn, nombre, texto_in):
        hallado = []

        def rep(m):
            dec = fn(m.group(1) if m.groups() else m.group(0))
            if dec:
                hallado.append(dec)
                return f" {dec} "
            return m.group(0)
        out = regex.sub(rep, texto_in)
        if hallado:
            r.tecnicas.append(nombre); r.decodificado.extend(hallado)
        return out

    t = _sub(RE_URLENC, lambda s: (lambda d: d if _legible(d, 3) else None)(unquote(s)), "url", t)
    t = _sub(RE_HEX, _hex, "hex", t)
    t = _sub(RE_BASE64, _b64, "base64", t)

    # Letras espaciadas
    largos: list[int] = []
    # Texto completamente espaciado («i g n o r a   l a s»): las palabras se separan por 2+ espacios
    if re.search(r"(?:\w ){4,}\w", t) and re.search(r"\s{2,}", t):
        segmentos = re.split(r"(\s{2,})", t)
        unidos = [re.sub(" ", "", g) if re.fullmatch(r"(?:\w )+\w", g) else g for g in segmentos]
        if unidos != segmentos:
            largos.append(max(len(g) for g in unidos))
            t = " ".join(g for g in unidos if g.strip())

    def unir(m):
        unido = re.sub(r"[ .\-_*·]", "", m.group(1))
        largos.append(len(unido))
        return unido
    t2 = RE_ESPACIADO.sub(unir, t)
    if largos and (max(largos) >= 5 or len(largos) >= 2):
        r.tecnicas.append("espaciado"); t = t2

    # Leetspeak: solo si afecta a 2+ tokens o a ≥30 % de las palabras (evita falsos positivos con «gpt4», «LLM01»)
    tokens = RE_TOKEN_LEET.findall(t)
    convertidos = {tok: c for tok in tokens if (c := _leet_token(tok))}
    if convertidos and (len(convertidos) >= 2 or len(convertidos) / max(len(tokens), 1) >= 0.3):
        t = RE_TOKEN_LEET.sub(lambda m: convertidos.get(m.group(0), m.group(0)), t)
        r.tecnicas.append("leetspeak")

    r.texto = re.sub(r"\s+", " ", t).strip()
    return r


# ---------------------------------------------------------------- generadores (para evaluar robustez)
def a_leet(texto: str) -> str:
    inverso = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    return "".join(inverso.get(c.lower(), c) for c in texto)


def a_espaciado(texto: str) -> str:
    return "   ".join(" ".join(p) for p in texto.split())


def a_base64(texto: str) -> str:
    return base64.b64encode(texto.encode("utf-8")).decode("ascii")


def a_homoglifos(texto: str) -> str:
    return texto.translate(str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}))


TRANSFORMACIONES = {"leetspeak": a_leet, "espaciado": a_espaciado, "base64": a_base64, "homoglifos": a_homoglifos}
