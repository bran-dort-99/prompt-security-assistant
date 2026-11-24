# Fase A — resultados (feedback del Avance 2)

Feedback: (1) detectar ofuscación (leetspeak, Base64); (2) ampliar ejemplos en español; (3) reforzar la frontera prompt_injection / jailbreak / system_prompt_leakage.

## Qué se implementó
1. **`src/nlp/ofuscacion.py`**: revierte ancho cero, homoglifos, letras espaciadas, leetspeak, Base64, hex y URL-encoding antes de las reglas y del ML; el chatbot y la cascada multimodal reportan las técnicas detectadas y el texto analizado. 17 pruebas (incluye 7 de falsos positivos: «LLM01», «gpt4», «4k», hashes, URLs).
2. **+221 ejemplos propios** (207 en español): frontera entre las tres categorías y negativos difíciles (`P_FINAL`). Guía: `docs/final/guia_etiquetado.md`.
3. **Prueba de frontera** escrita a mano (60 mensajes, ES/EN), excluida del entrenamiento.
4. **Chatbot**: las preguntas de seguimiento («¿a qué controles corresponde?») con evidencia de ataque débil usan el contexto de la conversación.

## Resultados (contenedor, sin deepset; `evidencias/final/`)
| Medida | Avance 2 | Final |
|---|---:|---:|
| Prueba de frontera — categoría exacta (híbrido, 45 ataques) | 0.36 | **0.60** |
| Prueba de frontera — ataque sí/no (híbrido, 60) | 0.72 | **0.77** |
| Prueba de frontera — falsos positivos en 15 benignas/consultas | 5 | **4** |
| Test interno — F1 `system_prompt_leakage` (mismo split; «Avance 2» = sin los datos nuevos) | 0.32 | **0.84** |
| Test interno — F1 `jailbreak` / `prompt_injection` (mismo split) | 0.61 / 0.80 | **0.66 / 0.85** |
| Test interno — F1 macro (mismo split, 540 ej.) | 0.685 | **0.734** |
| Batería — evasiones (híbrido) | 4/8 | **8/8** |
| Batería — directos / benignos (híbrido) | 18/18 · 8/10 | 18/18 · 8/10 |

**Advertencias.** (a) La batería de 36 prompts ya no es independiente: inspiró qué técnicas de ofuscación cubrir. Las pruebas independientes son la de frontera y deepset ofuscado. (b) El F1 macro 0.734 no es comparable con el 0.809 del Avance 2: el split cambió al añadir datos; la comparación justa es la de la tabla (ambos modelos en el mismo split). (c) Falta deepset (requiere red): correr `python -m src.nlp.evaluar_final` en el Mac.

**Errores que quedan** (prueba de frontera): el modelo ML falla las 5 inyecciones en inglés parafraseadas sin «ignore» (4 como benignas); jailbreaks por ficción o «fines educativos» en inglés; dos fugas del prompt confundidas con `data_leakage` («normas secretas»).
