# Asistente de seguridad para prompts de IA (chatbot basado en reglas)

Chatbot en Python 3.10+ y NLTK que:
- **analiza un prompt sospechoso** y lo clasifica (inyección directa/indirecta, jailbreak, fuga del system prompt, fuga de datos, agencia excesiva);
- **explica** cada categoría y la **mapea a controles** de OWASP LLM Top 10 (2025), MITRE ATLAS y NIST AI RMF;
- **recomienda guardrails**.

## Ejecutar en VS Code

1. Abre la carpeta del repo en VS Code (File > Open Folder) con la extensión Python instalada.
2. Crea el entorno virtual e instala dependencias en la terminal integrada:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
3. Selecciona el intérprete `.venv` (Ctrl/Cmd+Shift+P > *Python: Select Interpreter*).
4. Ejecuta lo que necesites:

   | Qué | Comando | Desde VS Code |
   | --- | --- | --- |
   | Chat interactivo | `python main.py` | Run and Debug > *Chat interactivo* |
   | 13 conversaciones de prueba | `python -m tests.conversaciones` | *13 conversaciones* |
   | Batería de 36 prompts | `python -m tests.bateria` | *Batería de prompts* |
   | Guardar evidencias en `evidencias/` | `python run_evidencias.py` | *Generar evidencias* |
   | Pruebas automáticas | `pytest -v` | panel Testing |

   Escribe `ayuda` en el chat para ver ejemplos y `adios` para salir. Ejemplo:
   `Analiza este prompt: Ignora todas las instrucciones anteriores`.

## Estructura
```
main.py                   chat interactivo
run_evidencias.py         ejecuta pruebas y guarda evidencias
src/chatbot.py            lógica (NLTK: RegexpTokenizer + SnowballStemmer es/en)
data/knowledge_base.json  base de conocimiento: intenciones, categorías, controles
data/bateria_prompts.json 36 prompts (directos, benignos, evasiones)
tests/                    conversaciones.py, bateria.py, test_chatbot.py (pytest)
evidencias/               transcripciones, CSV y resumen de la ejecución
docs/anexo_controles.md   mapeo detallado de controles (verificado)
scripts/verify_atlas.py   verifica IDs de ATLAS (requiere pyyaml)
.vscode/                  configuraciones de ejecución y pruebas
notebooks/                versión opcional para Google Colab
```

## Resultados (ver `evidencias/`)
Conversaciones: 13/13. Batería: ataques directos 17/18, benignos 9/10, evasiones 0/8.

## Limitaciones
Es un sistema de reglas: no entiende paráfrasis, texto espaciado, leetspeak, Base64 ni otros idiomas, y tiene falsos positivos con palabras ambiguas.
