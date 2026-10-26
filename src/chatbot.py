"""Asistente de seguridad para prompts de IA: chatbot basado en reglas con NLTK.

Flujo de cada mensaje:
    1. Normalización (minúsculas, sin acentos) y tokenización.
    2. Raíces de cada token con los stemmers Snowball de NLTK (español e inglés).
    3. Coincidencia de reglas de la base de conocimiento (data/knowledge_base.json).
    4. Selección de la intención y generación de la respuesta.

Uso rápido:
    from src.chatbot import AsistenteSeguridad
    bot = AsistenteSeguridad()
    print(bot.responder("Analiza este prompt: Ignora las instrucciones anteriores").texto)
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

KB_POR_DEFECTO = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

_ES = SnowballStemmer("spanish")
_EN = SnowballStemmer("english")
_TOKENIZADOR = RegexpTokenizer(r"\w+")

# Intenciones de cortesía: pierden frente a un ataque detectado en el mismo mensaje.
CORTESIA = {"saludo", "despedida", "agradecimiento"}
# Intenciones que dependen de una categoría (del mensaje o del contexto).
CON_CATEGORIA = {"consultar_categoria", "consultar_controles", "recomendar_guardrail"}

RE_OWASP = re.compile(r"\bllm\s?-?0?(\d{1,2})(?::\s?\d{4})?\b")
RE_ATLAS = re.compile(r"\baml\.?\s?([tm])(\d{4})(?:\.(\d{3}))?\b")
RE_NIST = re.compile(r"\b(measure|map|govern|manage)\s?(\d{1,2}\.\d{1,2})\b")
RE_CITA = re.compile(r"[«“\"](.+?)[»”\"]", re.S)
RE_DAN = re.compile(r"\bDAN\b")  # distingue el jailbreak «DAN» del verbo «dan»


def normalizar(texto: str) -> str:
    """Minúsculas y sin acentos (ñ pasa a n)."""
    base = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in base if not unicodedata.combining(c))


def raices(texto: str) -> set[str]:
    """Conjunto de raíces (español e inglés) de los tokens del texto."""
    salida: set[str] = set()
    for token in _TOKENIZADOR.tokenize(normalizar(texto)):
        salida.add(_ES.stem(token))
        salida.add(_EN.stem(token))
    return salida


def compilar(patron: str) -> list[set[str]]:
    """Convierte «a|b c» en grupos de raíces; el texto debe cumplir todos los grupos."""
    grupos = []
    for grupo in normalizar(patron).split():
        alternativas: set[str] = set()
        for palabra in grupo.split("|"):
            alternativas.add(_ES.stem(palabra))
            alternativas.add(_EN.stem(palabra))
        grupos.append(alternativas)
    return grupos


def mejor(reglas: dict[str, list[list[set[str]]]], raices_msg: set[str]) -> tuple[str | None, int]:
    """Clave cuya regla cumplida tiene más grupos. Empata el primero del diccionario."""
    mejor_clave, mejor_largo = None, 0
    for clave, lista in reglas.items():
        for regla in lista:
            if len(regla) > mejor_largo and all(grupo & raices_msg for grupo in regla):
                mejor_clave, mejor_largo = clave, len(regla)
    return mejor_clave, mejor_largo


@dataclass
class Respuesta:
    intencion: str
    texto: str
    categoria: str | None = None
    detectado: bool | None = None  # solo para analizar_prompt


class AsistenteSeguridad:
    def __init__(self, kb_path: Path | str = KB_POR_DEFECTO) -> None:
        self.kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self.categorias = self.kb["categorias"]
        self._r_intencion = {
            nombre: [compilar(p) for p in datos["patrones"]]
            for nombre, datos in self.kb["intenciones"].items() if datos["patrones"]
        }
        self._r_ataque = {c: [compilar(p) for p in d["patrones_ataque"]] for c, d in self.categorias.items()}
        self._r_consulta = {c: [compilar(p) for p in d["patrones_consulta"]] for c, d in self.categorias.items()}
        self.ultima_categoria: str | None = None  # contexto de la conversación

    # ---------- API pública ----------
    def reiniciar(self) -> None:
        self.ultima_categoria = None

    def reply(self, texto: str) -> str:
        return self.responder(texto).texto

    def responder(self, texto: str) -> Respuesta:
        texto = texto.strip()
        if not texto:
            return self._fija("fuera_de_alcance")
        norma = normalizar(texto)
        r_msg = raices(texto)

        # 1) Consulta de un control por su ID (LLM01, AML.T0051.001, MEASURE 2.7)
        control = self._control_por_id(norma)
        if control:
            return control

        # 2) Análisis explícito: «analiza este prompt: …»
        if self._cumple("analizar_prompt", r_msg):
            return self._analizar(self._extraer_prompt(texto))

        # 3) Intención informativa frente a ataque implícito
        intencion, largo = mejor({k: v for k, v in self._r_intencion.items() if k != "analizar_prompt"}, r_msg)
        ataque, largo_ataque = self._clasificar(texto, r_msg)
        categoria_msg = self._categoria_consulta(r_msg)

        if intencion in CORTESIA:
            return self._resultado_ataque(ataque) if ataque else self._fija(intencion)
        if intencion in CON_CATEGORIA:
            if categoria_msg is None and ataque:
                return self._resultado_ataque(ataque)
            return self._por_categoria(intencion, categoria_msg)
        if intencion:
            if ataque and largo_ataque > largo:
                return self._resultado_ataque(ataque)
            return self._informativa(intencion, r_msg)
        if ataque:
            return self._resultado_ataque(ataque)
        return self._fija("fuera_de_alcance")

    # ---------- detección ----------
    def _cumple(self, intencion: str, r_msg: set[str]) -> bool:
        return any(all(g & r_msg for g in regla) for regla in self._r_intencion[intencion])

    def _clasificar(self, texto: str, r_msg: set[str] | None = None) -> tuple[str | None, int]:
        r_msg = raices(texto) if r_msg is None else r_msg
        categoria, largo = mejor(self._r_ataque, r_msg)
        if RE_DAN.search(texto) and largo < 2:
            return "jailbreak", 2
        return categoria, largo

    def _categoria_consulta(self, r_msg: set[str]) -> str | None:
        return mejor(self._r_consulta, r_msg)[0]

    @staticmethod
    def _extraer_prompt(texto: str) -> str:
        cita = RE_CITA.search(texto)
        if cita:
            return cita.group(1)
        if ":" in texto:
            return texto.split(":", 1)[1].strip()
        return texto

    # ---------- respuestas ----------
    def _fija(self, intencion: str) -> Respuesta:
        return Respuesta(intencion, self.kb["intenciones"][intencion]["respuesta"])

    def _analizar(self, prompt: str) -> Respuesta:
        if not prompt:
            return Respuesta("analizar_prompt", "Pega el prompt que quieres analizar, por ejemplo: «Analiza este prompt: …».", detectado=False)
        categoria, _ = self._clasificar(prompt)
        if categoria is None:
            return Respuesta("analizar_prompt", self.kb["mensajes"]["sin_coincidencia"], detectado=False)
        return self._resultado_ataque(categoria)

    def _resultado_ataque(self, categoria: str) -> Respuesta:
        self.ultima_categoria = categoria
        datos = self.categorias[categoria]
        guardrails = "\n".join(f"{i}. {g}" for i, g in enumerate(datos["guardrails"][:2], 1))
        texto = (
            "Análisis del prompt\n"
            f"Categoría detectada: {datos['nombre']}.\n{datos['descripcion']}\n"
            f"Controles asociados:\n{self._lineas_controles(categoria)}\n"
            f"Guardrails sugeridos:\n{guardrails}\n"
            "Escribe «¿qué guardrail recomiendas?» para ver la lista completa."
        )
        return Respuesta("analizar_prompt", texto, categoria, True)

    def _lineas_controles(self, categoria: str) -> str:
        datos, ctl = self.categorias[categoria], self.kb["controles"]
        owasp = "; ".join(f"{ctl['owasp'][c]['id']} ({ctl['owasp'][c]['nombre']})" for c in datos["owasp"])
        atlas = "; ".join(f"{c} ({ctl['atlas'][c]['nombre']})" for c in datos["atlas"])
        nist = "; ".join(datos["nist"])
        return f"- OWASP LLM Top 10: {owasp}\n- MITRE ATLAS: {atlas}\n- NIST AI RMF: {nist}"

    def _por_categoria(self, intencion: str, categoria_msg: str | None) -> Respuesta:
        categoria = categoria_msg or self.ultima_categoria
        if categoria is None:
            clave = "sin_categoria" if intencion == "consultar_categoria" else "sin_contexto"
            lista = ", ".join(d["nombre"] for d in self.categorias.values())
            return Respuesta(intencion, self.kb["mensajes"][clave].format(lista=lista))
        self.ultima_categoria = categoria
        datos = self.categorias[categoria]
        if intencion == "recomendar_guardrail":
            pasos = "\n".join(f"{i}. {g}" for i, g in enumerate(datos["guardrails"], 1))
            texto = f"Guardrails recomendados contra «{datos['nombre']}»:\n{pasos}"
        elif intencion == "consultar_controles":
            texto = f"Controles asociados a «{datos['nombre']}»:\n{self._lineas_controles(categoria)}"
        else:
            texto = (
                f"{datos['nombre']}: {datos['descripcion']}\n"
                f"Controles asociados:\n{self._lineas_controles(categoria)}\n"
                "Pregúntame «¿qué guardrail recomiendas?» para ver cómo mitigarlo."
            )
        return Respuesta(intencion, texto, categoria)

    def _informativa(self, intencion: str, r_msg: set[str]) -> Respuesta:
        if intencion == "listar_categorias":
            filas = "\n".join(f"- {d['nombre']}: {d['descripcion']}" for d in self.categorias.values())
            return Respuesta(intencion, f"Conozco {len(self.categorias)} categorías de ataque:\n{filas}")
        if intencion == "consultar_framework":
            elegidos = [f for f, claves in (("nist", {"nist"}), ("owasp", {"owasp"}), ("atlas", {"atlas", "mitr"})) if claves & r_msg]
            elegidos = elegidos or ["nist", "owasp", "atlas"]
            return Respuesta(intencion, "\n".join(self.kb["frameworks"][f] for f in elegidos))
        return self._fija(intencion)

    def _control_por_id(self, norma: str) -> Respuesta | None:
        ctl, texto = self.kb["controles"], None
        if m := RE_ATLAS.search(norma):
            clave = f"AML.{m.group(1).upper()}{m.group(2)}" + (f".{m.group(3)}" if m.group(3) else "")
            dato = ctl["atlas"].get(clave)
            if dato:
                rel = [c["nombre"] for c in self.categorias.values() if clave in c["atlas"]]
                texto = f"{clave}: {dato['nombre']}. {dato['descripcion']}"
                texto += f"\nCategorías relacionadas aquí: {', '.join(rel)}." if rel else ""
        elif m := RE_OWASP.search(norma):
            clave = f"LLM{int(m.group(1)):02d}"
            dato = ctl["owasp"].get(clave)
            if dato:
                rel = [c["nombre"] for c in self.categorias.values() if clave in c["owasp"]]
                texto = f"{dato['id']}: {dato['nombre']}. {dato['descripcion']}"
                texto += f"\nCategorías relacionadas aquí: {', '.join(rel)}." if rel else ""
        elif m := RE_NIST.search(norma):
            clave = f"{m.group(1).upper()} {m.group(2)}"
            dato = ctl["nist"].get(clave)
            if dato:
                rel = [c["nombre"] for c in self.categorias.values() if clave in c["nist"]]
                texto = f"NIST AI RMF {clave}: {dato['descripcion']}"
                texto += f"\nCategorías relacionadas aquí: {', '.join(rel)}." if rel else ""
        else:
            return None
        return Respuesta("consultar_control", texto or self.kb["mensajes"]["control_desconocido"])


def chatear() -> None:
    """Bucle de consola. En Colab se usa bot.responder(...) directamente."""
    bot = AsistenteSeguridad()
    print("Asistente de seguridad para prompts de IA. Escribe «ayuda» o «adios».")
    while True:
        try:
            texto = input("Tú: ")
        except EOFError:
            break
        respuesta = bot.responder(texto)
        print(f"Asistente: {respuesta.texto}")
        print()  # línea en blanco para separar un turno del siguiente
        if respuesta.intencion == "despedida":
            break


if __name__ == "__main__":
    chatear()
