"""Preprocesamiento de texto (D6): cuatro técnicas explícitas y separadas.

tokenizar -> normalizar -> quitar_stopwords (con lista blanca) -> stem.
NLTK no tiene lematizador para español, por eso se usa stemming Snowball es+en.
"""
from __future__ import annotations

import re
import unicodedata

from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

_ES = SnowballStemmer("spanish")
_EN = SnowballStemmer("english")
_TOKENIZADOR = RegexpTokenizer(r"\w+")

# Palabras que NLTK trata como stopwords pero que importan en ataques.
LISTA_BLANCA = {
    "you", "your", "all", "not", "no", "now", "me", "my",
    "tu", "tus", "todo", "todas", "todos", "ahora", "sin",
}

# Respaldo cuando el corpus de NLTK no está disponible (p. ej. sin internet).
_RESPALDO_ES = (
    "a al algo ante como con contra cual cuando de del desde donde el ella ellos en entre era es esa ese eso esta este "
    "esto fue ha han hay la las le les lo los mas me mi mis muy ni o os otra otro para pero por porque que se si sin "
    "sobre su sus te ti tu tus un una uno unos y ya yo nos"
).split()
_RESPALDO_EN = (
    "a about above after again against all am an and any are as at be because been before being below between both but "
    "by can did do does doing down during each few for from further had has have having he her here hers him his how i "
    "if in into is it its just me more most my no nor not now of off on once only or other our out over own same she so "
    "some such than that the their them then there these they this those through to too under until up very was we were "
    "what when where which while who whom why will with you your yours"
).split()


def _cargar_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        palabras = set(stopwords.words("spanish")) | set(stopwords.words("english"))
    except (LookupError, OSError):
        palabras = set(_RESPALDO_ES) | set(_RESPALDO_EN)
    return {normalizar(p) for p in palabras}


def normalizar(texto: str) -> str:
    """Minúsculas y sin acentos (ñ pasa a n)."""
    base = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in base if not unicodedata.combining(c))


_RE_URL = re.compile(r"(?:https?://|www\.)\S+")
_RE_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def normalizar_texto(texto: str) -> str:
    """Normalización completa del clasificador: minúsculas, sin acentos, URL y números a marcadores."""
    texto = _RE_URL.sub(" xurlx ", texto)
    texto = normalizar(texto)
    return _RE_NUM.sub(" xnumx ", texto)


def tokenizar(texto: str) -> list[str]:
    return _TOKENIZADOR.tokenize(texto)


_STOP: set[str] | None = None


def quitar_stopwords(tokens: list[str], lista_blanca: bool = True) -> list[str]:
    global _STOP
    if _STOP is None:
        _STOP = _cargar_stopwords()
    protegidas = LISTA_BLANCA if lista_blanca else set()
    return [t for t in tokens if t not in _STOP or t in protegidas]


def stem(tokens: list[str]) -> list[str]:
    """Raíz Snowball: inglés si el token no cambia con el stemmer español y viceversa; se prefiere la más corta."""
    salida = []
    for t in tokens:
        if t in ("xurlx", "xnumx"):
            salida.append(t)
            continue
        salida.append(min(_ES.stem(t), _EN.stem(t), key=len))
    return salida


def preprocesar(texto: str, *, stopwords: bool = True, stemming: bool = True, lista_blanca: bool = True) -> list[str]:
    tokens = tokenizar(normalizar_texto(texto))
    if stopwords:
        tokens = quitar_stopwords(tokens, lista_blanca)
    if stemming:
        tokens = stem(tokens)
    return tokens


def demo_pasos(texto: str) -> dict[str, object]:
    """Cada etapa por separado (tabla «antes/después» del informe)."""
    norm = normalizar_texto(texto)
    tok = tokenizar(norm)
    sin = quitar_stopwords(tok)
    return {"original": texto, "normalizado": norm, "tokens": tok, "sin_stopwords": sin, "stems": stem(sin)}
