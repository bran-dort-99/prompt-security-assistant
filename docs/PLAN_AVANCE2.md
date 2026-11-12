# Plan de implementación — Avance de proyecto 2

> **Para quien implementa (Sonnet, después de compactar):** este archivo es la fuente de verdad. Léelo completo antes de escribir código. Implementa por fases, en orden, y marca cada casilla al terminar. No cambies las decisiones de la sección 2 sin consultar a Brandon.

- **Repo:** `github.com/bran-dort-99/prompt-security-assistant` (local: `/home/claude/prompt-security-assistant`, clonado de forma anónima; **no hay permiso de push** desde esta sesión: Brandon debe conectar GitHub o subir los cambios él mismo).
- **Entrega:** PDF, **domingo 15-nov-2026, 23:59 (CDMX)**. Pondera 15 %.
- **Rúbrica (100 pts):** NLP 20 · clasificador de texto 20 · CNN 25 · integración 20 · análisis 15.
- **Base existente (Avance 1):** chatbot basado en reglas con NLTK (`src/chatbot.py`), `data/knowledge_base.json` (11 intenciones conversacionales, 6 categorías de ataque, controles OWASP/ATLAS/NIST verificados), `data/bateria_prompts.json` (36 prompts: 18 directos, 10 benignos, 8 evasiones), 13 conversaciones y pruebas con pytest. Resultado base de las reglas: **directos 17/18, benignos 9/10, evasiones 0/8**. Este es el baseline contra el que compite el modelo de ML.

---

## 1. Restricciones del entorno (verificadas el 11-nov)

| Recurso | Workspace de Claude | Google Colab |
|---|---|---|
| PyPI (`pip install`) | ✅ | ✅ |
| Hugging Face datasets / hub | ❌ bloqueado | ✅ |
| Pesos de torchvision (`download.pytorch.org`) | ❌ bloqueado | ✅ |
| Kaggle | ❌ | ✅ (requiere token) |
| GPU | ❌ (2 CPU, 7 GB) | ✅ T4 |

**Consecuencias:**
- En el workspace: escribir el código, el generador sintético, las pruebas y hacer **pruebas de humo** (CNN con `weights=None`, 2 épocas, decenas de imágenes; texto con muestras propias).
- **El entrenamiento real (descarga de datasets de HF, fine-tuning de la CNN) se hace en Colab.** Los notebooks deben poder ejecutarse de principio a fin con `git clone` del repo + celdas en orden.
- Evitar dependencias de sistema: OCR con **EasyOCR** (solo pip) y decodificación de QR con **`cv2.QRCodeDetector`** (OpenCV, solo pip). No usar pyzbar ni Tesseract.

---

## 2. Decisiones de diseño

| # | Decisión | Por qué |
|---|---|---|
| D1 | **Clases de la CNN: `foto_natural`, `texto_incrustado`, `codigo_qr`.** La CNN detecta el *vector* (si hay instrucciones o un código dentro de la imagen); la *intención* la decide OCR/QR → clasificador de texto. | Una CNN a 224 px no lee: no distingue "texto malicioso" de "texto benigno" con la misma tipografía (lo confirman el reporte de investigación y FigStep-Pro). Con estas clases, las métricas de la CNN son significativas y la cascada reutiliza el componente de NLP, lo que fortalece la integración. **⚠ Esto cambia lo propuesto antes (`benign / visual_prompt_injection / qr_or_encoded`). Confirmar con Brandon antes de la Fase 3.** |
| D2 | **Experimento B (control):** un segundo entrenamiento corto con las clases `texto_benigno` vs `texto_malicioso` (misma tubería de render). Se espera un desempeño cercano al azar. | Es evidencia empírica para el análisis: justifica la cascada CNN → OCR → NLP y da la limitación "sustentada en resultados" que pide la rúbrica. |
| D3 | **Backbone principal: EfficientNet-B0** (`torchvision.models.efficientnet_b0`, `EfficientNet_B0_Weights.IMAGENET1K_V1`); **ResNet50** (`ResNet50_Weights.IMAGENET1K_V2`) como comparación si queda tiempo. | B0 entrena rápido en una T4 y sus pesos (~16 MB) caben en el repo; ResNet50 (~95 MB) no. |
| D4 | **Intenciones del clasificador de texto (12 etiquetas):** conversacionales `saludo`, `despedida`, `agradecimiento`, `ayuda`, `consulta_seguridad`, `benigno`; de ataque `prompt_injection`, `indirect_injection`, `jailbreak`, `system_prompt_leakage`, `data_leakage`, `excessive_agency`. | Cubre las intenciones ya definidas en la KB (la rúbrica exige "usar las intenciones definidas para el asistente"). `consulta_seguridad` agrupa `listar_categorias`, `consultar_categoria`, `consultar_controles`, `recomendar_guardrail` y `consultar_framework`; el sub-ruteo entre ellas sigue con reglas (son casi léxicas y tienen pocos datos). `benigno` reemplaza a `fuera_de_alcance`. |
| D5 | **Chatbot híbrido:** el clasificador de ML decide la intención; las reglas se mantienen para (a) la búsqueda de controles por ID (`LLM01`, `AML.T0051.001`), (b) la extracción del prompt citado, (c) el sub-ruteo de `consulta_seguridad` y (d) como respaldo cuando la confianza del modelo es menor que el umbral (`UMBRAL = 0.45`, ajustable). | Conserva lo que ya funciona y permite comparar reglas vs ML con la misma batería. |
| D6 | **Preprocesamiento (las 4 técnicas, explícitas y separadas):** tokenización `nltk.RegexpTokenizer(r"\w+")` → normalización (minúsculas, sin acentos, reemplazo de URLs/números por `<url>`/`<num>`) → stopwords `nltk.corpus.stopwords` es+en **con una lista blanca** de palabras que importan en ataques (`you, your, all, not, no, now, me, my, tu, tus, todo, todas, ahora, no, sin`) → **stemming** Snowball es+en (NLTK no tiene lematizador para español; documentarlo como la razón para elegir stemming). | Cumple el criterio de 20 pts y reutiliza las funciones del Avance 1. La lista blanca es un hallazgo para el reporte (ablación). |
| D7 | **Representación:** comparar BoW (`CountVectorizer`) vs TF-IDF de palabras (1–2 gramas sobre los tokens preprocesados) vs **TF-IDF palabras + TF-IDF de caracteres `char_wb` 3–5** (`FeatureUnion`). **Clasificadores:** `MultinomialNB`/`ComplementNB` vs `LinearSVC` (envuelto en `CalibratedClassifierCV` para obtener probabilidades). Se elige uno por F1 macro en validación. | Los n-gramas de caracteres ayudan contra la ofuscación (leetspeak, espaciado), que es justo donde las reglas sacaron 0/8. |
| D8 | **Evaluación honesta:** (1) test estratificado del dataset unificado; (2) **test externo:** `deepset/prompt-injections` (split test, 116 filas) excluido del entrenamiento, evaluado colapsando a ataque/benigno; (3) **la batería de 36 prompts del Avance 1**, comparando reglas vs ML por tipo (directo/benigno/evasión). | La rúbrica premia el análisis por intención; el test externo y la comparación con las reglas son la evidencia más fuerte. |
| D9 | **Artefactos:** el modelo de texto (`models/clasificador_intenciones.joblib`) y la CNN B0 (`models/cnn_efficientnet_b0.pt`, si pesa < 25 MB) se versionan. Los datasets crudos y las imágenes generadas **no** se versionan (`.gitignore`), solo el código que los produce, los CSV propios y un `manifest.csv` con el origen de cada fila. | Reproducibilidad sin inflar el repo; los datasets de terceros tienen licencias variadas. |

---

## 3. Estructura objetivo del repo

```
src/
  chatbot.py                 # MODIFICAR: modo híbrido (D5) + comando de imagen
  nlp/
    __init__.py
    preprocesamiento.py      # tokenizar, normalizar, quitar_stopwords, stem, preprocesar (D6)
    dataset_texto.py         # descarga/mapeo de datasets públicos + propios → data/intenciones/dataset.csv
    clasificador_texto.py    # construir_pipeline(), entrenar(), evaluar(), cargar(), predecir()
  vision/
    __init__.py
    generar_imagenes.py      # generador sintético con PIL + qrcode (sección 5)
    modelo.py                # construir_modelo(backbone, n_clases), transforms, cargar(), predecir()
    entrenar.py              # bucle de entrenamiento (congelado → fine-tune), métricas, figuras
  multimodal.py              # AnalizadorImagen: CNN → OCR/QR → clasificador de texto → resultado
data/
  intenciones/
    propios_es_en.csv        # ejemplos escritos/plantillas (versionado)
    dataset.csv              # unificado: texto, etiqueta, fuente, idioma, split (generado)
  imagenes/                  # .gitignore
  imagenes_reales/           # fotos de Brandon (versionar si pesan poco) + manifest.csv
models/
notebooks/
  avance2_01_nlp.ipynb       # Colab CPU
  avance2_02_cnn.ipynb       # Colab GPU
  avance2_03_integracion.ipynb
tests/
  test_preprocesamiento.py
  test_clasificador_texto.py
  test_vision_smoke.py       # weights=None, 3 clases, pocas imágenes
  test_multimodal.py
evidencias/avance2/          # métricas JSON/CSV, matrices de confusión PNG, curvas, transcripciones
docs/avance2/informe.md      # borrador del documento → PDF
```

**`requirements.txt` (agregar):** `scikit-learn>=1.4`, `pandas`, `joblib`, `matplotlib`, `pillow`, `qrcode[pil]`, `opencv-python-headless`, `datasets` (solo para descargar en Colab), `torch`, `torchvision`, `easyocr`. Separar en `requirements-ml.txt` si `torch` hace demasiado pesado el entorno de VS Code.

---

## 4. Componente de texto (NLP)

### 4.1 Dataset (`src/nlp/dataset_texto.py` → `data/intenciones/dataset.csv`)

| Etiqueta | Fuentes | Objetivo de filas |
|---|---|---|
| `prompt_injection` | `xTRam1/safe-guard-prompt-injection` (label 1, muestrear), `deepset/prompt-injections` **train** (label 1) | 300 |
| `jailbreak` | `jackhhao/jailbreak-classification` (`jailbreak`), JBB-Behaviors | 300 |
| `system_prompt_leakage` | filtrar inyecciones de xTRam1/deepset con regex (`system prompt|instructions above|initial prompt|reveal your (rules|prompt)`, también en español) + plantillas propias es/en | 150 |
| `data_leakage` | plantillas propias (credenciales, PII, API keys, datos de clientes, base de datos) es/en + filtrado por regex | 150 |
| `excessive_agency` | plantillas propias (borrar archivos, transferir dinero, enviar correos, ejecutar comandos sin confirmación) es/en | 150 |
| `indirect_injection` | plantillas propias: instrucciones dentro de "documento/correo/página/imagen" (`<!-- ignore… -->`, "texto en blanco", "[nota del sistema en el PDF]") es/en | 150 |
| `benigno` | xTRam1 (label 0), deepset train (label 0), `jackhhao` (`benign`), JBB benignos (hard negatives) | 400 |
| `saludo`, `despedida`, `agradecimiento`, `ayuda`, `consulta_seguridad` | plantillas propias es/en a partir de los patrones de `knowledge_base.json` y de las conversaciones del Avance 1 | 80 c/u |

- Los ejemplos propios se escriben en `propios_es_en.csv` con columnas `texto,etiqueta,idioma,plantilla_id`. Pueden generarse con ayuda de IA: **declararlo en el informe**.
- **Partición:** test estratificado del 20 %; para los ejemplos generados con plantillas, usar `GroupShuffleSplit` por `plantilla_id` para que la misma plantilla no aparezca en train y test.
- **Nunca** incluir `deepset` test ni `bateria_prompts.json` en el entrenamiento (deduplicar por texto normalizado).
- Guardar en `evidencias/avance2/dataset_resumen.csv` la tabla de filas por etiqueta × fuente × idioma (va al PDF).

### 4.2 Preprocesamiento (`src/nlp/preprocesamiento.py`)
Funciones puras y testeables: `tokenizar`, `normalizar`, `quitar_stopwords`, `stem`, `preprocesar(texto) -> list[str]`. Reutilizar `normalizar` y los stemmers de `src/chatbot.py` (moverlos aquí y que `chatbot.py` los importe). Incluir una función `demo_pasos(texto)` que devuelva cada etapa: el PDF necesita una tabla "antes/después" con 3–4 ejemplos.

### 4.3 Experimentos y métricas (`src/nlp/clasificador_texto.py`, notebook 01)
1. Matriz de experimentos: {BoW, TF-IDF palabras, TF-IDF palabras+chars} × {ComplementNB, LinearSVC calibrado}, con validación cruzada estratificada de 5 folds sobre train → tabla de F1 macro.
2. Ablación sobre el mejor modelo: sin quitar stopwords / sin stemming / sin la lista blanca.
3. Modelo final → test: `classification_report` por intención, matriz de confusión 12×12 (PNG), F1 macro y ponderado.
4. Test externo `deepset` (ataque vs benigno): P/R/F1.
5. Batería de 36: ML vs reglas por tipo → tabla. Guardar los errores para el análisis.
6. Guardar `models/clasificador_intenciones.joblib` + `evidencias/avance2/texto_metricas.json`.

---

## 5. Componente visual (CNN)

### 5.1 Dataset (`src/vision/generar_imagenes.py`)
Salida: `data/imagenes/{train,val,test_sintetico}/{foto_natural,texto_incrustado,codigo_qr}/*.jpg` + `manifest.csv` (archivo, clase, split, fuente, fuente_tipografica, fondo, texto_origen, params).

- **Fotos base:** COCO `val2017` (5,000 imágenes, `http://images.cocodataset.org/zips/val2017.zip`, se descarga en Colab). Usar ~1,500 como `foto_natural` y otras ~1,500 **distintas** como fondos. Separar los IDs de COCO por split para evitar fugas.
- **`texto_incrustado` (~1,500):** renderizar textos del dataset de texto (**mezcla de ataque y benignos**, porque la clase es el vector, no la intención; D1). Variar: ≥8 fuentes TTF (descargar de Google Fonts en Colab; en el workspace, usar `fc-list`/DejaVu para la prueba de humo), tamaño, color, fondo (blanco, color sólido, foto COCO, ruido), estilo (pasos numerados tipo FigStep, Post-it, captura de chat, letrero), rotación ±15°, **bajo contraste (ΔRGB ≤ 25) en ~15 %**, compresión JPEG y desenfoque.
- **`codigo_qr` (~1,500):** generar con `qrcode` los payloads (URLs benignas, URLs sospechosas, texto de inyección, Base64), pegados sobre fondos COCO/sólidos con perspectiva, escala y rotación aleatorias; ~20 % de códigos de barras (opcional, `python-barcode`). Guardar el payload en el manifest (sirve para la integración).
- **Anti-atajos:** el mismo pool de fondos y el mismo rango de tamaño final (redimensionar todo a 256 px por el lado menor) en todas las clases; `foto_natural` incluye fotos COCO que ya contienen letreros (aceptar el ruido de etiqueta y mencionarlo).
- **Generalización:** `test_sintetico` usa **fuentes y fondos no vistos** en train.
- **Test real** (`data/imagenes_reales/`): (a) imágenes de FigStep/SafeBench (`git clone https://github.com/CryptoAILab/FigStep` en Colab) → `texto_incrustado`; (b) MM-SafetyBench TYPO si la descarga es viable (opcional; CC BY-NC 4.0); (c) **~30 fotos con el celular de Brandon:** pantallas con prompts, notas Post-it, QR reales (menú, pago), fotos sin texto. Etiquetarlas en `imagenes_reales/manifest.csv`.

### 5.2 Modelo y entrenamiento (`src/vision/modelo.py`, `entrenar.py`, notebook 02)
- Transforms: train `RandomResizedCrop(224, scale=(0.7,1))`, `ColorJitter` leve, **sin flip horizontal** (invierte el texto y los QR); val/test `Resize(256)` + `CenterCrop(224)`; normalización de ImageNet.
- Etapa 1: backbone congelado, nueva cabeza `Dropout(0.3)+Linear(n,3)`, AdamW lr 1e-3, 5 épocas. Etapa 2: descongelar los últimos 2 bloques, lr 1e-4, hasta 10 épocas con early stopping (paciencia 3) sobre el F1 macro de val. CrossEntropy con `class_weight` si hay desbalance. Fijar semillas.
- Métricas sobre `test_sintetico` **y** el test real: accuracy, precision/recall/F1 por clase y macro, matriz de confusión (PNG), curvas de pérdida y accuracy. Guardar `evidencias/avance2/cnn_metricas.json` y una grilla de errores (las 12 imágenes peor clasificadas).
- Experimento B (D2): mismo código, clases `texto_benigno` vs `texto_malicioso`, solo etapa 1 + 3 épocas. Reportar accuracy y la matriz.
- (Opcional) ResNet50 con el mismo protocolo → tabla comparativa.

---

## 6. Integración multimodal (`src/multimodal.py` + `chatbot.py`)

Flujo de `AnalizadorImagen.analizar(ruta) -> ResultadoImagen`:
1. CNN → clase + probabilidades.
2. `codigo_qr` → `cv2.QRCodeDetector().detectAndDecode()` → si hay payload: clasificador de texto + regla de URL (esquema, IP literal, acortadores, `data:`/`javascript:`).
3. `texto_incrustado` **o** confianza < 0.6 → EasyOCR (`['es','en']`, inicialización perezosa y cacheada) → clasificador de texto.
4. `foto_natural` con confianza alta → sin riesgo (aun así, OCR si `--estricto`).
5. Decisión: si el texto extraído se clasifica como ataque → la categoría del ataque con **vector = imagen/QR**; la respuesta reutiliza `_resultado_ataque` y agrega una sección "Evidencia visual" (clase de la CNN y su probabilidad, texto extraído truncado, payload del QR). Si es una inyección en imagen, forzar también `indirect_injection` (AML.T0051.001, LLM01) como en la KB.

**Chatbot:** nuevo patrón `analiza (la|esta) imagen: <ruta>` / `analyze image: <ruta>` (regex antes de la clasificación de intenciones). En Colab, función `chat_con_imagen(texto, ruta_imagen)` y una celda con `files.upload()`. Mantener `responder()` compatible: las 13 conversaciones del Avance 1 deben seguir pasando (o actualizar las expectativas justificando cada cambio).

**Evidencia requerida por la rúbrica (3 operaciones):** notebook 03 con al menos **6 casos de extremo a extremo**: texto benigno, texto de ataque, foto natural, imagen con inyección tipográfica, imagen con texto benigno (sin falso positivo), QR con payload malicioso. Guardar la transcripción en `evidencias/avance2/integracion_transcripcion.txt` + capturas.

---

## 7. Pruebas (pytest, deben pasar en el workspace sin GPU ni internet)
- `test_preprocesamiento.py`: cada etapa por separado; la lista blanca conserva `you/todas/ignora`.
- `test_clasificador_texto.py`: entrenar con `propios_es_en.csv` (mínimo) y predecir etiquetas válidas con probabilidades que sumen 1.
- `test_vision_smoke.py`: generar 10 imágenes por clase, `construir_modelo(weights=None)`, 1 época, forma de la salida `(B,3)`.
- `test_multimodal.py`: un QR generado se decodifica y su payload malicioso termina en categoría de ataque (CNN simulada con un *stub*).
- Las pruebas del Avance 1 siguen en verde.

---

## 8. Documento PDF (`docs/avance2/informe.md` → PDF)
Secciones obligatorias en este orden: Introducción · Conjunto de datos textual · Proceso de preprocesamiento de texto · Representación y clasificación de intenciones · Conjunto de imágenes · Arquitectura y desarrollo del clasificador visual · Integración multimodal · Resultados · Métricas · Conclusiones · Referencias.

Contenido mínimo para "Altamente competente":
- **Introducción:** recuperar el problema, objetivo, usuarios, alcance y arquitectura del Avance 1, más un diagrama de la arquitectura actualizada (texto → NLP; imagen → CNN → OCR/QR → NLP → mapeo de controles), señalando lo que se corrigió.
- **Resultados/Métricas:** tablas de la sección 4.3 y 5.2, las matrices de confusión, ML vs reglas en la batería, el test externo y el experimento B.
- **Análisis por intención:** qué intenciones se confunden entre sí (se espera `prompt_injection`↔`jailbreak`↔`system_prompt_leakage`) y por qué.
- **Limitaciones sustentadas en los números:** caída en el test externo y/o en el test real vs sintético; la CNN no lee (experimento B); datos mayormente en inglés vs chatbot en español; el OCR es evadible (FigStep-Pro, bajo contraste); datos de plantillas que inflan las métricas.
- **Conclusiones → acciones para la versión final:** web (Avance 3), más datos reales en español, umbral/calibración, CLIP u OCR robusto, base de datos de hallazgos.
- **Referencias APA:** datasets (deepset, xTRam1, jackhhao, JBB, COCO, FigStep, MM-SafetyBench), papers (FigStep arXiv 2311.05608, MM-SafetyBench ECCV 2024, Kornblith et al. 1805.08974, EfficientNet, ResNet), OWASP 2025, ATLAS v2026.09, NIST AI 100-1 / 600-1, scikit-learn, NLTK, PyTorch, EasyOCR.

---

## 9. Orden de trabajo y checklist

**Fase 0 — Preparación (workspace)**
- [x] Confirmar D1 con Brandon. *(confirmado 11-nov: se acepta foto_natural / texto_incrustado / codigo_qr)*
- [x] Rama `avance-2`; crear la estructura; actualizar requirements y `.gitignore` (`data/imagenes/`, `data/intenciones/raw/`, `models/*.pt` > 25 MB).
- [x] Mover las utilidades de NLP de `chatbot.py` a `src/nlp/preprocesamiento.py` sin romper las pruebas.

**Fase 1 — NLP (workspace + Colab, jue 12)**
- [x] `propios_es_en.csv` (todas las etiquetas propias, es/en).
- [x] `dataset_texto.py` (descarga en Colab; en el workspace, funcionar solo con los propios).
- [x] `clasificador_texto.py` + pruebas.
- [ ] Notebook 01 en Colab → métricas, figuras, `.joblib` → commit.

**Fase 2 — Chatbot híbrido (jue 12)**
- [ ] Integrar el clasificador en `responder()` (D5); conversaciones y batería en verde; tabla ML vs reglas.

**Fase 3 — CNN (vie 13)**
- [ ] `generar_imagenes.py` + prueba de humo local.
- [ ] Notebook 02 en Colab (GPU): generar, entrenar B0, evaluar sintético + real, experimento B → commit de métricas, figuras y pesos.
- [ ] Brandon toma las ~30 fotos reales (pedírselas temprano).

**Fase 4 — Integración (sáb 14)**
- [ ] `multimodal.py` + comando en el chat + pruebas.
- [ ] Notebook 03 con los 6 casos de extremo a extremo + evidencias.

**Fase 5 — Documento (sáb 14 – dom 15)**
- [ ] `informe.md` con las 11 secciones → PDF. Revisar contra la rúbrica, criterio por criterio.
- [ ] Actualizar el README (cómo ejecutar el Avance 2).
- [ ] Verificación final: `pytest -v` en verde, los notebooks corren de principio a fin en Colab, las cifras del PDF coinciden con `evidencias/avance2/*.json`.

---

> Nota Fase 0: en el workspace `nltk.download('stopwords')` está bloqueado; `preprocesamiento.py` usa el corpus si existe y una lista de respaldo si no. En Colab se usa el corpus real (reportar cuál se usó en el informe).

> Nota Fase 1: la CV del plan se hace **agrupada por `plantilla_id`**; sin agrupar daba F1 0.96 vs 0.56 en test (fuga de plantillas). Documentar en el informe. Notebook 01 creado; falta ejecutarlo en Colab y commitear sus artefactos.

## 10. Riesgos y planes B
| Riesgo | Plan B |
|---|---|
| HF/COCO lento o caído en Colab | Para texto, usar solo los propios + deepset; para imágenes, usar `torchvision.datasets.Caltech101` o `Flowers102` como fotos. |
| Las métricas de la CNN salen ~100 % (atajo) | Revisar la grilla de errores; endurecer el render (más fondos foto, bajo contraste); el número que importa es el del test real. |
| EasyOCR demasiado lento en CPU | Usar `gpu=True` en Colab; en local, limitar el lado mayor a 1,024 px. |
| Las conversaciones del Avance 1 cambian de comportamiento | El umbral de respaldo a reglas (D5) se sube; documentar los casos que cambian. |
| Sin tiempo para ResNet50 | Basta con EfficientNet-B0: la rúbrica pide "ResNet50 **o** EfficientNet". |
