# Anexo A. Controles de referencia para el mapeo de hallazgos

Proyecto: red teaming automatizado para chatbots (NIST AI RMF, OWASP LLM Top 10, MITRE ATLAS).
Fecha de corte: 20 de octubre de 2026.

Este anexo vive en el repositorio y no forma parte del reporte. Sirve como fuente del módulo `app/mapping/controls.py` y de la tabla `control_mapping` de la base de datos.

## 1. Versiones y estado de verificación

| Fuente | Versión usada | Verificación |
| --- | --- | --- |
| MITRE ATLAS | v2026.09 (15-sep-2026), formato de datos 6.0.0 | Verificada contra `ATLAS-2026.09.yaml` del repositorio oficial con `scripts/verify_atlas.py` (IDs, nombres, tácticas y vínculos técnica→mitigación) |
| NIST AI RMF 1.0 (AI 100-1) | enero de 2023; en revisión por el Plan de Acción de IA de EE. UU., sin texto revisado publicado | Investigación sobre el documento oficial; no se reverificó con script |
| NIST AI 600-1 (perfil de IA generativa) | julio de 2024 | Investigación sobre el documento oficial; no se reverificó con script |
| OWASP Top 10 for LLM Applications | edición 2025 (18-nov-2024) | Investigación sobre el documento oficial |
| OWASP GenAI LLM Top 10 | edición 2026 (agosto de 2026) | LLM01:2026 leído en el repositorio oficial; orden y renombres por resúmenes de terceros |
| OWASP Top 10 for Agentic Applications | edición 2026 (diciembre de 2025), ASI01 a ASI10 | Comunicado oficial de OWASP |

Convención: guardar siempre el ID junto con la edición o versión (`LLM01:2025`, `AML.T0051.001` con ATLAS `2026.09`, `MS-2.7-007` con `AI 600-1 (2024-07)`). OWASP renumeró categorías en 2026 y ATLAS publica una versión cada mes.

## 2. Mapeo ataque → controles

Dos ejes separados: la categoría del ataque (`attack_category`) y el vector de entrega (`delivery_vector`: texto, imagen, archivo o salida de herramienta). En OWASP, imagen y archivo no tienen categoría propia: son vectores de LLM01.

| Categoría | OWASP 2025 (2026) | MITRE ATLAS 2026.09 | NIST AI RMF / AI 600-1 | Guardrail típico |
| --- | --- | --- | --- | --- |
| (a) Inyección de instrucciones directa e indirecta | LLM01:2025 (LLM01:2026); agentes: ASI01 | AML.T0051.000 directa, AML.T0051.001 indirecta, AML.T0051.002 diferida | MEASURE 2.7 (MS-2.7-007, MS-2.7-004); MAP 5.1 (MP-5.1-005); MAP 2.3 (MP-2.3-005); MANAGE 3.2 (MG-3.2-005) | Clasificador de inyección en la entrada; etiquetar y aislar contenido externo; validar el formato de salida; mínimo privilegio |
| (b) Jailbreak | LLM01:2025 (subtipo, según el texto oficial) | AML.T0054; con ofuscación, AML.T0068 | MEASURE 2.6 (MS-2.6-006, MS-2.6-007) primaria; MEASURE 2.7 (MS-2.7-007) secundaria; MS-2.5-006 | Moderación de entrada y salida; normalizar y decodificar (Base64, Unicode, idiomas) antes del filtro; refuerzo de alineación |
| (c) Fuga de datos sensibles o PII | LLM02:2025 (LLM02:2026) | AML.T0057; extracción desde entrenamiento: AML.T0024.000 y AML.T0024.001 | MEASURE 2.10 (MS-2.10-001) primaria; MAP 4.1 (MP-4.1-001, MP-4.1-009); MEASURE 2.2 (MS-2.2-002) | Detectar y redactar PII en la salida; control de acceso previo a la recuperación (RAG con permisos); minimizar datos |
| (d) Fuga del system prompt | LLM07:2025 (LLM08:2026, «Hidden Context Exposure») | AML.T0056; descubrimiento: AML.T0069.002 | MEASURE 2.7 (MS-2.7-001, MS-2.7-004); MEASURE 2.10 solo si el prompt contiene PII | No poner secretos en el prompt; tokens canario y comparación de salida contra el prompt; autorización fuera del LLM |
| (e) Inyección indirecta por imagen con texto incrustado | LLM01:2025 (escenario de inyección multimodal) | AML.T0051.001; AML.T0129 «Triggers in Multimodal Inputs» | MEASURE 2.7 (MS-2.7-007); MAP 4.1 (MP-4.1-009 incluye imagen); MANAGE 3.2 (MG-3.2-005) | OCR y luego el mismo clasificador de inyección que en texto; clasificador de imágenes; recodificar o reescalar; tratar el texto extraído como no confiable |
| (f) Inyección indirecta por archivos (PDF o DOCX con instrucciones ocultas) | LLM01:2025; si el archivo entra a un RAG, también LLM08:2025 | AML.T0051.001; si persiste en RAG: AML.T0070 y AML.T0071; ocultamiento: AML.T0068 | MEASURE 2.7 (MS-2.7-007); MANAGE 3.1 (MG-3.1-001); MAP 4.1 y 4.2; GOVERN 6.1 | Sanitizar el archivo: extraer texto plano, descartar texto oculto o del color del fondo, metadatos, comentarios, macros y Unicode invisible; escanear antes de indexar |
| (g) Exceso de agencia o abuso de herramientas | LLM06:2025 (LLM03:2026); agentes: ASI02 y ASI03 | AML.T0053; AML.T0086; AML.T0101; herramienta comprometida: AML.T0110 y AML.T0010.005 | MEASURE 2.7 (MS-2.7-001); MAP 3.5; GOVERN 3.2; MANAGE 2.4 (MG-2.4-002) | Mínimo privilegio por herramienta; lista blanca de herramientas; aprobación humana en acciones irreversibles; mediación completa en un motor de políticas; límites de tasa y registro |

Los vínculos de ATLAS entre técnicas y mitigaciones están en la sección 5.

## 3. Mapeo para el código y esquema de base de datos

`app/mapping/controls.py` (sustituye al mapeo preliminar del borrador):

```python
CONTROL_MAP = {
    "prompt_injection": {
        "owasp": "LLM01:2025",
        "atlas": ["AML.T0051.000", "AML.T0051.001"],
        "nist": ["MEASURE 2.7"],
    },
    "jailbreak": {
        "owasp": "LLM01:2025",
        "atlas": ["AML.T0054"],
        "nist": ["MEASURE 2.6", "MEASURE 2.7"],
    },
    "data_leakage": {
        "owasp": "LLM02:2025",
        "atlas": ["AML.T0057"],
        "nist": ["MEASURE 2.10", "MAP 4.1"],
    },
    "system_prompt_leakage": {
        "owasp": "LLM07:2025",
        "atlas": ["AML.T0056"],
        "nist": ["MEASURE 2.7"],
    },
    "excessive_agency": {
        "owasp": "LLM06:2025",
        "atlas": ["AML.T0053", "AML.T0086"],
        "nist": ["MEASURE 2.7", "MAP 3.5"],
    },
}

# Técnicas ATLAS adicionales según el vector de entrega
VECTOR_ATLAS = {
    "text": [],
    "image": ["AML.T0051.001", "AML.T0129"],
    "file": ["AML.T0051.001"],
    "tool_output": ["AML.T0051.001"],
}
```

Tabla `control_mapping`, un renglón por control asignado a un hallazgo:

| Columna | Ejemplo |
| --- | --- |
| finding_id | 1042 |
| framework | `MITRE-ATLAS`, `OWASP-LLM`, `NIST-AI-600-1` |
| framework_version | `2026.09`, `2025`, `2024-07` |
| control_id | `AML.T0051.001`, `LLM01:2025`, `MS-2.7-007` |
| control_name | `Indirect`, `Prompt Injection`, … |
| role | `primary` o `secondary` |
| verified_source | `atlas-data v2026.09` |
| verified_date | `2026-10-20` |

Corrección frente al mapeo preliminar del borrador:

| Clave | Borrador | Corrección |
| --- | --- | --- |
| prompt_injection | LLM01, AML.T0051, MEASURE 2.7 | Añadir la edición y la subtécnica: LLM01:2025 y AML.T0051.000 (directa) o .001 (indirecta) |
| jailbreak | LLM01, AML.T0054, MEASURE 2.7 | NIST: MEASURE 2.6 como primaria y 2.7 como secundaria |
| data_leakage | LLM02, AML.T0057, MEASURE 2.10 | Correcto; añadir MAP 4.1 y la edición de OWASP |
| system_prompt_leakage | no existía | Clave nueva: LLM07:2025, AML.T0056, MEASURE 2.7 |

Errores frecuentes en fuentes secundarias: numerar la inyección directa e indirecta como .001 y .002 (lo correcto es .000 y .001; .002 es «Triggered»), llamar a AML.T0054 «Prompt Injection: Indirect», y usar «LLM Meta Prompt Extraction» como nombre vigente de AML.T0056. OWASP 2025 cita AML.T0051.000 para la fuga del system prompt; esa referencia está desactualizada.

## 4. NIST AI RMF 1.0 y perfil AI 600-1

NIST no publica un crosswalk oficial entre ataques y subcategorías. La asignación primaria y secundaria de este anexo es una recomendación razonada. El perfil AI 600-1 no incluye todas las subcategorías del AI RMF.

### Subcategorías del AI RMF

| ID | Qué exige | Uso en el proyecto |
| --- | --- | --- |
| GOVERN 1.5 | Monitoreo continuo y revisión periódica del manejo de riesgos | Cadencia de las campañas de red teaming |
| GOVERN 3.2 | Políticas que definen roles en la relación humano-IA | Aprobación humana de políticas y acciones de agentes |
| GOVERN 4.3 | Prácticas que habilitan pruebas de IA e identificación de incidentes | Justifica la existencia del chatbot atacante |
| GOVERN 6.1 | Riesgos de terceros | Chatbots objetivo y plugins de terceros |
| MAP 2.3 | Integridad científica y consideraciones de pruebas (TEVV) documentadas | Diseño experimental |
| MAP 3.5 | Procesos de supervisión humana definidos y documentados | Exceso de agencia |
| MAP 4.1 y 4.2 | Riesgos de componentes y controles internos documentados | Inventario de herramientas, RAG y archivos |
| MAP 5.1 | Probabilidad y magnitud de cada impacto documentadas | Severidad de hallazgos |
| MEASURE 1.1 | Métricas para los riesgos más significativos | Tasa de éxito de ataque (ASR) |
| MEASURE 2.1 | Conjuntos de prueba, métricas y herramientas documentados | Base de datos de hallazgos |
| MEASURE 2.6 | El sistema se evalúa con regularidad frente a riesgos de seguridad | Jailbreak y evasión de salvaguardas |
| MEASURE 2.7 | Seguridad y resiliencia evaluadas y documentadas | Inyección, fuga del system prompt, abuso de herramientas |
| MEASURE 2.10 | Riesgo de privacidad examinado y documentado | Fuga de PII |
| MEASURE 3.1 | Seguimiento de riesgos existentes y emergentes | Regresión entre campañas |
| MEASURE 4.2 | Resultados validados en el contexto de despliegue | Validación de hallazgos con expertos |
| MANAGE 1.3 | Respuestas a riesgos de alta prioridad planificadas y documentadas | Recomendación de guardrails |
| MANAGE 2.4 | Mecanismos para desactivar sistemas | Interruptor de apagado |
| MANAGE 3.1 y 3.2 | Monitoreo de riesgos de terceros y modelos preentrenados | Filtros sobre recursos de terceros |

### Acciones sugeridas de AI 600-1 usadas (paráfrasis)

| Acción | Contenido |
| --- | --- |
| MS-2.7-007 | Red teaming de resiliencia frente a abuso para atacar otros sistemas, ataques a IA generativa (como prompt injection) y ataques a ML. Acción ancla para inyección y jailbreak |
| MS-2.7-001 | Medidas de seguridad contra amenazas como puertas traseras, dependencias comprometidas, agentes autónomos y extracción |
| MS-2.7-004 | Métricas de efectividad de controles: intentos de acceso no autorizado, evasión, extracción |
| MS-2.7-006 | Tasa de implementación de las recomendaciones de controles |
| MS-2.6-006 y MS-2.6-007 | Verificar el manejo de consultas de uso malicioso; evaluar con regularidad la vulnerabilidad a eludir medidas de seguridad |
| MS-2.5-006 | Revisar con regularidad los guardrails de seguridad |
| MP-2.3-005 | Pruebas adversariales regulares para encontrar vulnerabilidades y manipulación |
| MP-5.1-005 | Juegos de rol adversariales, red teaming y chaos testing para hallar fallos imprevistos |
| MS-2.8-002 | Documentar las instrucciones dadas a los red-teamers (aplica a los prompts del chatbot atacante) |
| MS-4.2-001 | Pruebas adversariales con cadencia regular |
| MG-1.3-002 | Monitorear la robustez de los controles mediante red teaming |
| MG-3.1-001 y MG-3.2-005 | Controles sobre recursos de terceros, incluido el filtrado de entradas y salidas con reglas o modelos adicionales |
| MG-2.4-002 y MG-2.4-004 | Escalamiento y criterios de desactivación |
| MG-4.3-001 y GV-4.3-002 | Revisión posterior a incidentes; campos mínimos de reporte (ID del sistema, título, quien reporta, fecha, descripción, impacto), útiles como esquema de la tabla de hallazgos |
| MS-2.10-001 | Red teaming de privacidad: salida de muestras de entrenamiento, extracción, inferencia de pertenencia y revelación de información confidencial o personal |
| MP-4.1-001 y MP-4.1-009 | Monitorear la salida por riesgos de privacidad; detectar PII en texto, imagen, video o audio generados |
| MS-2.2-002 y MS-2.2-004 | Filtros de privacidad en salidas, eliminación de PII, anonimización y privacidad diferencial |

## 5. OWASP

### Top 10 for LLM Applications, edición 2025

| ID | Nombre | Mitigaciones oficiales principales |
| --- | --- | --- |
| LLM01:2025 | Prompt Injection | Restringir el comportamiento en el system prompt; validar formatos de salida; filtrar entrada y salida; mínimo privilegio; aprobación humana; aislar contenido externo; pruebas adversariales |
| LLM02:2025 | Sensitive Information Disclosure | Sanitizar datos; validar entradas; control de acceso estricto; privacidad diferencial; tokenización y redacción |
| LLM03:2025 | Supply Chain | Verificar proveedores; SBOM; firmas y hashes de modelos; red teaming al elegir modelos de terceros |
| LLM04:2025 | Data and Model Poisoning | Rastrear el origen de los datos; sandboxing; versionado de datos; red teaming |
| LLM05:2025 | Improper Output Handling | Tratar al modelo como usuario no confiable; codificar la salida según el contexto; consultas parametrizadas |
| LLM06:2025 | Excessive Agency | Minimizar extensiones y permisos; evitar extensiones abiertas; ejecutar en el contexto del usuario; aprobación humana; mediación completa |
| LLM07:2025 | System Prompt Leakage | No considerar el prompt un secreto ni un control de seguridad; separar datos sensibles; guardrails externos al LLM |
| LLM08:2025 | Vector and Embedding Weaknesses | Control de acceso en la base vectorial; validar fuentes; extracción de texto que detecte contenido oculto |
| LLM09:2025 | Misinformation | RAG con fuentes verificadas; supervisión humana; validación automática |
| LLM10:2025 | Unbounded Consumption | Validar tamaño de entrada; límites de tasa y cuotas; tiempos máximos; monitoreo de recursos |

### Edición 2026 (para trabajo futuro)

Orden de la edición 2026: LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM03 Excessive Agency, LLM04 Supply Chain, LLM05 Data and Model Poisoning, LLM06 Unbounded Consumption, LLM07 Misinformation, LLM08 Hidden Context Exposure, LLM09 Vector and Embedding Weaknesses, LLM10 Improper Output Handling. Cambios que afectan a este proyecto: Excessive Agency pasa de LLM06 a LLM03, y System Prompt Leakage (LLM07:2025) se convierte en Hidden Context Exposure (LLM08:2026). LLM01:2026 extiende la inyección a imagen, audio, video, razonamiento intermedio y memoria persistente, y recomienda clasificadores y OCR por modalidad, además de eliminar caracteres Unicode invisibles. La fuente canónica es el repositorio `GenAI-Security-Project/GenAI-LLM-Top10`, carpeta `2026/final`.

### Top 10 for Agentic Applications, edición 2026

ASI01 Agent Goal Hijack, ASI02 Tool Misuse and Exploitation, ASI03 Identity and Privilege Abuse, ASI04 Agentic Supply Chain Vulnerabilities, ASI05 Unexpected Code Execution, ASI06 Memory and Context Poisoning, ASI07 Insecure Inter-Agent Communication, ASI08 Cascading Failures, ASI09 Human-Agent Trust Exploitation, ASI10 Rogue Agents. Para este proyecto importan ASI01 (complementa LLM01) y ASI02 y ASI03 (complementan LLM06).

## 6. MITRE ATLAS v2026.09 (verificado)

Resultado de `scripts/verify_atlas.py` el 20 de octubre de 2026 contra `ATLAS-2026.09.yaml`: 18 técnicas y 13 mitigaciones comprobadas, sin discrepancias. El catálogo contiene 16 tácticas, 120 técnicas, 88 subtécnicas y 40 mitigaciones.

### Técnicas

| ID | Nombre | Táctica(s) | Madurez | Mitigaciones vinculadas en ATLAS |
| --- | --- | --- | --- | --- |
| AML.T0051 | LLM Prompt Injection | Execution | Realized | M0019, M0020, M0021, M0022, M0024, M0033, M0035 |
| AML.T0051.000 | Direct | Execution | Realized | M0024, M0033, M0035 |
| AML.T0051.001 | Indirect | Execution | Demonstrated | M0024, M0033, M0035 |
| AML.T0051.002 | Triggered | Execution | Demonstrated | M0024, M0033, M0035 |
| AML.T0054 | LLM Jailbreak | Defense Evasion, Privilege Escalation | Realized | M0020, M0021, M0022, M0035 |
| AML.T0056 | Extract LLM System Prompt | Exfiltration | Feasible | M0020, M0021, M0022, M0035 |
| AML.T0057 | LLM Data Leakage | Exfiltration | Demonstrated | M0008, M0020, M0021, M0022, M0035 |
| AML.T0068 | LLM Prompt Obfuscation | Defense Evasion | Realized | M0020, M0035 |
| AML.T0069.002 | System Prompt (bajo Discover LLM System Information) | Discovery | Demonstrated | M0005, M0019, M0020 |
| AML.T0053 | AI Agent Tool Invocation | Execution, Privilege Escalation, Lateral Movement | Demonstrated | M0020, M0021, M0022, M0024, M0026 a M0030, M0032, M0033, M0035 |
| AML.T0086 | Exfiltration via AI Agent Tool Invocation | Exfiltration | Realized | M0024, M0026 a M0030, M0032, M0033 |
| AML.T0101 | Data Destruction via AI Agent Tool Invocation | Impact | Realized | M0024, M0026 a M0030 |
| AML.T0110 | AI Agent Tool Poisoning | Persistence | Realized | ninguna vinculada |
| AML.T0010.005 | AI Agent Tool (bajo AI Supply Chain Compromise) | Initial Access | Realized | M0035 |
| AML.T0070 | RAG Poisoning | Persistence | Demonstrated | M0020, M0035 |
| AML.T0071 | False RAG Entry Injection | Defense Evasion | Demonstrated | M0020 |
| AML.T0077 | LLM Response Rendering | Exfiltration | Demonstrated | ninguna vinculada |
| AML.T0129 | Triggers in Multimodal Inputs | Defense Evasion | Feasible | ninguna vinculada |

AML.T0129 es nueva en v2026.09. Describe instrucciones o disparadores colocados en una modalidad que los controles no inspeccionan (imagen, audio, video o metadatos de archivo). Encaja con las categorías (e) y (f). Como ATLAS aún no vincula mitigaciones a esta técnica, el recomendador de guardrails debe tomar como referencia las de AML.T0051.001.

### Mitigaciones

| ID | Nombre |
| --- | --- |
| AML.M0008 | Validate AI Model |
| AML.M0020 | Generative AI Guardrails |
| AML.M0021 | Generative AI Guidelines |
| AML.M0022 | Generative AI Model Alignment |
| AML.M0024 | AI Telemetry Logging |
| AML.M0026 | Privileged AI Agent Permissions Configuration |
| AML.M0027 | Single-User AI Agent Permissions Configuration |
| AML.M0028 | AI Agent Tools Permissions Configuration |
| AML.M0029 | Human In-the-Loop for AI Agent Actions |
| AML.M0030 | Restrict AI Agent Tool Invocation on Untrusted Data |
| AML.M0032 | Segmentation of AI Agent Components |
| AML.M0033 | Input and Output Validation for AI Agent Components |
| AML.M0035 | AI Red Team |

Para actualizar la verificación en cada campaña: `python scripts/verify_atlas.py --fetch`. El archivo `dist/v6/ATLAS-latest.yaml` del repositorio oficial solo contiene el nombre del archivo vigente; el script lo resuelve solo.

## 7. Limitaciones

- NIST y OWASP se tomaron de la investigación sobre los documentos oficiales y no se reverificaron con script. Conviene revisar los IDs de acciones de AI 600-1 contra el PDF antes de la entrega final.
- El orden y los renombres de OWASP 2026 provienen de resúmenes de terceros; solo LLM01:2026 se leyó en el repositorio oficial.
- La revisión del AI RMF puede cambiar la numeración; verificar antes de la defensa.
- Los vínculos técnica→mitigación de ATLAS son los que publica el catálogo; técnicas sin vínculos (T0110, T0077, T0129) no implican que no existan mitigaciones, solo que ATLAS aún no las registra.

## 8. Referencias

Instituto Nacional de Estándares y Tecnología. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

Instituto Nacional de Estándares y Tecnología. (2024). *Artificial intelligence risk management framework: Generative artificial intelligence profile* (NIST AI 600-1). https://doi.org/10.6028/NIST.AI.600-1

MITRE Corporation. (2026). *ATLAS data* (versión de contenido v2026.09; formato 6.0.0) [Conjunto de datos]. GitHub. https://github.com/mitre-atlas/atlas-data

OWASP GenAI Security Project. (2024). *OWASP top 10 for LLM applications 2025*. OWASP Foundation. https://genai.owasp.org/llm-top-10/

OWASP GenAI Security Project. (2025). *OWASP top 10 for agentic applications for 2026*. OWASP Foundation. https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/

OWASP GenAI Security Project. (2026). *OWASP GenAI LLM top 10 2026* [Archivo Markdown]. GitHub. https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/tree/main/2026/final
