# Asistente de seguridad para prompts de IA

**Avance 1:** chatbot basado en reglas. **Avance 2:** NLP + clasificador de intenciones (ML) + CNN de imágenes + cascada multimodal (rama `avance-2`).

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


## Avance 2 (NLP, CNN e integración)

Informe: `docs/avance2/informe.pdf` (fuente: `informe.md`; se reconstruye con `bash scripts/construir_informe.sh`). Evidencias, JSON y capturas: `evidencias/avance2/`.

```bash
python -m venv prompt && source prompt/bin/activate
pip install -r requirements-ml.txt pytest
python -m pytest -q                              # 57 pruebas
python -m src.nlp.dataset_texto                  # arma data/intenciones/dataset.csv (descarga datasets públicos)
python -m src.nlp.clasificador_texto             # entrena, evalúa, guarda models/clasificador_intenciones.joblib
python -m tests.bateria --comparar               # reglas vs ML vs híbrido
python -m src.vision.generar_imagenes --fondos data/coco/val2017   # imágenes sintéticas (COCO val2017)
python -m src.vision.entrenar --workers 2        # CNN EfficientNet-B0; --solo-reales evalúa sin reentrenar
```
Notebooks: `notebooks/avance2_01_nlp.ipynb`, `avance2_02_cnn.ipynb`, `avance2_03_integracion.ipynb` (VS Code o Colab).
Chat con imagen: `analiza esta imagen: ruta/foto.jpg`.

**Resultados:** clasificador de intenciones F1 macro 0.809 (prueba interna), F1 0.835 en deepset; CNN 0.915 de exactitud en prueba sintética y 0.773 en 22 fotos reales; cascada 0.864 de exactitud de riesgo en las mismas fotos.

**Datos y declaraciones:** los ejemplos propios usan plantillas redactadas con ayuda de IA; 9 de las 22 imágenes reales de evaluación son de ataque generadas con IA (`fuente=generada_ia`). Las fotos con datos personales no se publican (`data/imagenes_reales/privadas/` está en `.gitignore`).
