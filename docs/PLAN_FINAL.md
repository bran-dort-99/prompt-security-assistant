# Plan — Proyecto final (Semana 8, 40 %)

> Fuente de verdad del proyecto final. Repo: `bran-dort-99/prompt-security-assistant`, rama `final` (parte de `avance-2`).
> **Fecha límite supuesta: domingo 6-dic-2026, 23:59 CDMX** (Semana 8 = 30-nov a 6-dic). Confirmar en la plataforma.
> Semana 8 se reserva para revisar, integrar y entregar: **todo lo funcional debe estar listo el domingo 29-nov.**

## Feedback del Avance 2 (profesor)
1. Detectar ofuscación (leetspeak y Base64) → **hecho** (Fase A).
2. Ampliar ejemplos en español → **hecho** (Fase A).
3. Reforzar la frontera prompt_injection / jailbreak / system_prompt_leakage → **hecho** (Fase A); quedan errores en inglés parafraseado.

## 0. Decisiones (confirmar o cambiar antes del martes 24)

| ID | Decisión | Propuesta | Por qué |
|---|---|---|---|
| F1 | Dónde corre la inferencia | **API Flask en el PC con RTX 4060 (WSL2)**, expuesta con **Cloudflare Tunnel**; el Mac M3 queda como respaldo | torch + EasyOCR + modelo (~3 GB de dependencias) no caben en el plan gratuito de Streamlit Community Cloud (~1 GB de RAM) |
| F2 | Dónde corre la interfaz | **Streamlit Community Cloud** (URL pública `*.streamlit.app`), cliente ligero que llama a la API | Es el recurso del proyecto base; no necesita GPU |
| F3 | Túnel | **Named tunnel** con dominio propio si tienes uno en Cloudflare; si no, *quick tunnel* (`trycloudflare.com`) y la URL se actualiza en `st.secrets` antes de la demo | El *quick tunnel* cambia de URL en cada arranque |
| F4 | Tipo de agente | **Híbrido**: capa reactiva (guardrails y reglas inmediatas) + capa deliberativa (planificador que elige herramientas y encadena pasos) | Encaja con lo que ya existe: reglas del Avance 1 = reactivo, cascada multimodal = plan de varios pasos |
| F5 | ¿LLM dentro del agente? | **No en el núcleo.** Planificador determinista y explicable. LLM solo como mejora futura | Sin costo, reproducible, más fácil de justificar en ética y en pruebas |
| F6 | Mejoras de modelos | **Feedback del profesor (obligatorio):** detección de ofuscación (leetspeak/Base64), más español, frontera prompt_injection / jailbreak / system_prompt_leakage. Además: QR sin fondo en el sintético y modo `--estricto` evaluado. ResNet50 solo si sobra tiempo | Lo pidió el profesor y ataca las limitaciones medidas en el Avance 2 |

## 1. Arquitectura objetivo

```
Usuario ──► Streamlit (Community Cloud) ──HTTPS + API key──► Cloudflare Tunnel ──► Flask API (PC 4060 / Mac)
                │  modo degradado si la API no responde                         │
                │                                                               ▼
                │                                                     Agente híbrido
                │                                         ┌───────────────┴────────────────┐
                │                                   Capa reactiva                  Capa deliberativa
                │                              (límites, guardrails,           (planificador → herramientas)
                │                               reglas rápidas)                       │
                │                                                    ┌────────────┬──────┴─────┬─────────────┬──────────────┐
                │                                               classify_text  analyze_image  decode_qr/ocr  map_controls  build_report
                │                                               (TF-IDF+SVM)   (EfficientNet)  (OpenCV/EasyOCR) (KB OWASP/ATLAS/NIST)
                └──────────── /health, /metrics ◄── monitoreo (logs JSON, latencias, errores)
```

**Flujo de coordinación del agente (percibir → decidir → actuar → informar):**
1. **Percibir**: entrada = texto, imagen o ambos; estado de sesión (historial, hallazgos previos).
2. **Capa reactiva** (sin planificar): tamaño/tipo de archivo, saludo/ayuda/fuera de alcance, consulta directa de un control por ID → responde de inmediato.
3. **Capa deliberativa**: arma un plan según la entrada (p. ej. imagen → `analyze_image` → si QR `decode_qr` → `classify_text` sobre el *payload* → `map_controls` → `build_report`). Re-planifica si una herramienta falla o la confianza es baja (p. ej. CNN < 0.60 → QR + OCR; OCR vacío → aviso).
4. **Actuar e informar**: respuesta con categoría, controles, guardrails, evidencia y **traza del plan** (pasos, herramienta, tiempo, confianza) visible en la UI.

## 2. API REST (Flask)

| Método y ruta | Uso |
|---|---|
| `GET /health` | Estado y versión de modelos (sin API key) |
| `POST /v1/agent` | Punto principal: texto y/o imagen → plan + respuesta |
| `POST /v1/classify/text` | Intención + probabilidades |
| `POST /v1/classify/image` | Clase CNN + probabilidad (+ OCR/QR opcional) |
| `GET /v1/controls/<id>` | Consulta de un control (LLM01, AML.T0051…) |
| `GET /metrics` | Conteos, errores, latencia p50/p95 por ruta |

Requisitos: API key en cabecera (`X-API-Key`), límite de tamaño (5 MB) y tipos (jpg/png/webp), *rate limit* simple por IP, `request_id` en cada respuesta, errores JSON uniformes (`400/401/413/415/429/500/503`), modelos cargados una sola vez al arrancar, imágenes **no se guardan** (se procesan en memoria y se elimina EXIF).

## 3. Interfaz (Streamlit)

Páginas: **Asistente** (chat + subir imagen, traza del agente plegable, tarjetas de controles), **Analizar imagen** (clase, texto OCR, payload QR, señales de URL), **Métricas** (texto, CNN, cascada, chatbot desde `evidencias/`), **Monitoreo** (`/health` y `/metrics` de la API), **Acerca de / Ética** (alcance, límites, privacidad, aviso de uso responsable).
UX del dominio: lenguaje de analista de seguridad, severidad con color, botón de copiar guardrails, ejemplos de un clic, aviso claro cuando la API está caída (modo degradado: solo reglas del Avance 1 en el propio Streamlit).

## 4. Checklist por fases

### Fase A — Corrección y base (lun 23 – mar 24)
- [ ] Confirmar decisiones F1–F6 y la fecha límite real.
- [ ] Rama `final` desde `avance-2`; merge de `avance-2` a `main` para que el repo público muestre el trabajo.
- [ ] Recuperar y corregir: problema, objetivo, usuarios, alcance, arquitectura (incluir la corrección «la CNN clasifica el vector, no la intención»).
- [x] Normalización de ofuscación (`src/nlp/ofuscacion.py`: leetspeak, espaciado, Base64, hex, URL, homoglifos, ancho cero) + 17 pruebas. *(feedback)*
- [x] +221 ejemplos propios (207 en español) y reentreno; guía de etiquetado de la frontera. *(feedback)*
- [x] Prueba de frontera escrita a mano (60) + `src/nlp/evaluar_final.py` (Avance 2 vs final). Resultados: `docs/final/resultados_fase_a.md`.
- [ ] Correr `python -m src.nlp.evaluar_final` y `python -m src.nlp.clasificador_texto` en el Mac (con red) para deepset y deepset ofuscado.

### Fase B — Agente + API (mié 25 – jue 26)
- [ ] `src/agente/`: estado, registro de herramientas, capa reactiva, planificador, re-planificación, traza.
- [ ] `api/app.py` (Flask): rutas, API key, validaciones, manejo de errores, logs JSON, `/metrics`.
- [ ] Pruebas: unitarias del agente, `pytest` de la API con el *test client*, casos de error.
- [ ] QR sin fondo en el generador sintético + reentrenar CNN (≈5 min) + evaluar modo `--estricto` en reales.

### Fase C — Web + despliegue (vie 27 – dom 29)
- [ ] `app/streamlit_app.py` con las 5 páginas; cliente HTTP con *timeouts*, reintentos y modo degradado.
- [ ] Levantar la API en el PC (WSL2) con `gunicorn` + `cloudflared`; servicio que arranque solo.
- [ ] Desplegar en Streamlit Community Cloud con `API_URL` y `API_KEY` en *secrets*.
- [ ] Capturas de despliegue (Cloudflare, Streamlit Cloud, URL pública funcionando).

### Fase D — Pruebas y análisis (lun 30 – mié 2)
- [ ] Matriz de pruebas funcionales de la app integrada (≥20 casos: texto, imagen, QR, errores, API caída).
- [ ] Prueba de carga pequeña (p. ej. 50–200 peticiones concurrentes) → latencia p50/p95, errores, uso de GPU/CPU.
- [ ] Métricas finales: clasificación (texto, CNN, cascada) y chatbot (conversaciones, batería, tasa de respuesta fuera de alcance).
- [ ] Análisis de desempeño, escalabilidad y mantenibilidad.

### Fase E — Documento, video y entrega (jue 3 – dom 6)
- [ ] Documento técnico PDF de 15–20 cuartillas (13 secciones, abajo).
- [ ] Diagrama de arquitectura final (+ diagrama del agente).
- [ ] Video de 5–10 min (guion abajo).
- [ ] Verificación final: URL pública abre, enlace del repo funciona, `pytest` en verde, cifras del PDF = JSON.
- [ ] Entrega en la plataforma (PDF, enlace al repo, enlace al video, URL pública) antes del domingo 6, idealmente el sábado 5.

## 5. Documento técnico (15–20 cuartillas)

| Sección | Contenido | Págs. | Reutiliza |
|---|---|---:|---|
| Introducción | Contexto, objetivo, qué cambió desde los avances | 1 | Avance 1–2 |
| Descripción del problema | Problema, usuarios, alcance, beneficios (corregidos) | 1 | Avance 1 |
| Arquitectura general | Diagrama final, componentes, flujo de datos | 1.5 | nuevo |
| Módulo de NLP | Datos, preprocesamiento, TF-IDF + SVM, normalización de ofuscación, métricas | 2 | Avance 2 §2–4 |
| Módulo de visión | Datos, EfficientNet-B0, métricas sintético/real, QR sin fondo | 2 | Avance 2 §5–6 |
| Arquitectura del agente | Tipo híbrido, capas, herramientas, flujo de coordinación, re-planificación | 2 | nuevo |
| Desarrollo web | Páginas, UX del dominio, manejo de errores | 1.5 | nuevo |
| API REST | Rutas, contratos, seguridad, errores, monitoreo | 1.5 | nuevo |
| Despliegue en la nube | Streamlit Cloud + Cloudflare Tunnel + GitHub, capturas | 1.5 | nuevo |
| Resultados | Pruebas funcionales, métricas de clasificación y chatbot, carga, desempeño/escalabilidad/mantenibilidad | 2.5 | nuevo + Avance 2 |
| Consideraciones éticas | Ver §7 | 1.5 | nuevo |
| Conclusiones | Logros, limitaciones, **plan de mejoras futuras** | 1 | nuevo |
| Referencias | APA | 1 | Avance 2 + Flask, Streamlit, Cloudflare |

## 6. Guion del video (≈8 min)
1. (0:00–0:45) Problema y usuarios.
2. (0:45–2:00) Arquitectura en el diagrama: Streamlit → Cloudflare → Flask → agente → modelos.
3. (2:00–3:30) Demo texto: inyección directa, evasión en otro idioma, consulta de un control por ID; mostrar la traza del agente.
4. (3:30–5:30) Demo imagen: captura con inyección, QR malicioso, foto benigna; mostrar OCR/QR y la re-planificación por baja confianza.
5. (5:30–6:30) Robustez: API caída → modo degradado; archivo inválido → error claro; página de monitoreo.
6. (6:30–7:30) Métricas y limitaciones.
7. (7:30–8:00) Ética y mejoras futuras.

## 7. Ética (puntos a desarrollar)
- **Doble uso**: el asistente describe ataques; se limita a detectar, explicar y recomendar controles, sin generar *payloads* nuevos.
- **Privacidad**: imágenes en memoria, sin retención, EXIF eliminado, logs sin contenido del usuario (solo metadatos).
- **Errores y su costo**: falsos negativos (ataque no detectado) vs. falsos positivos (fricción); umbrales justificados y comunicados; el sistema apoya, no sustituye al analista (*human-in-the-loop*).
- **Sesgo lingüístico**: 86 % de los datos en inglés → peor desempeño en español; medido y declarado.
- **Transparencia**: traza del agente visible; datos generados con IA declarados; límites en la página «Acerca de».
- **Seguridad propia**: API key, límites de tamaño y tasa, sin ejecutar contenido de URL ni QR.
- Marco: NIST AI RMF (Govern/Map/Measure/Manage), OWASP LLM Top 10 2025.

## 8. Entregables → evidencia

| Entregable | Dónde |
|---|---|
| PDF 15–20 cuartillas | `docs/final/informe.pdf` → `Prompt-Security-Final-BDOP.pdf` |
| Repo con enlace funcional | `main` (tras merge) |
| Video 5–10 min | YouTube no listado / Drive con enlace público |
| URL pública | `https://<app>.streamlit.app` |
| Diagrama de arquitectura final | `docs/final/figuras/` |
| Evidencias de pruebas | `evidencias/final/pruebas_*.csv`, salida de `pytest` |
| Capturas de despliegue | `evidencias/final/capturas/` |
| Métricas de clasificación | `evidencias/final/*_metricas.json` |
| Métricas del chatbot | `evidencias/final/chatbot_metricas.json` |
| Plan de mejoras futuras | sección de Conclusiones + `docs/final/mejoras.md` |

## 9. Riesgos

| Riesgo | Mitigación |
|---|---|
| El PC/Mac se apaga o duerme durante la revisión → URL rota | Servicio con arranque automático, `caffeinate`/energía siempre encendido la semana de entrega; **modo degradado** en Streamlit; video como evidencia de respaldo |
| URL del *quick tunnel* cambia | Named tunnel, o actualizar `API_URL` en *secrets* y verificar justo antes de entregar |
| Dependencias pesadas en Streamlit Cloud | La UI no importa torch/EasyOCR; solo `requests`, `streamlit`, `pandas` |
| Falta de tiempo | Recortes en este orden: ResNet50 → prueba de carga grande → páginas extra de la UI |

## 10. Pendiente de Brandon
- Confirmar F1–F6 y la fecha límite.
- ¿Tienes un dominio en Cloudflare (para el *named tunnel*)?
- Pegar la **rúbrica** del proyecto final cuando esté disponible, para mapear criterio por criterio.
