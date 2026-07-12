# QuantAsistente

**Asistente experto en finanzas cuantitativas** — RAG · Google Gemini · ChromaDB · LangGraph

Proyecto Final del módulo de IA Generativa.

Un agente conversacional que responde preguntas sobre finanzas cuantitativas leyendo **su propia
base de conocimiento vectorial**, construida a partir de **5 documentos académicos reales** de MIT,
Columbia University e IARE. Cada respuesta indica **de qué documento y de qué página** sale, y el
agente **recuerda** la conversación: puedes preguntarle _«¿y sus limitaciones?»_ y sabe de qué le
hablas.

**🔗 App desplegada → [https://TU-APP.streamlit.app](https://proyecto-rag-nihalhk.streamlit.app/)**
**📓 Documentación completa → [`asistente_rag_gemini.ipynb`](asistente_rag_gemini.ipynb)**

---

> ### 📌 Dónde está cada cosa
>
> **El notebook es autosuficiente**: contiene el código, la documentación, la justificación de cada
> decisión de diseño y los ejemplos ya ejecutados. Este README es solo la puerta de entrada.
>
> | Si buscas…                                           | Ve a                                   |
> | ---------------------------------------------------- | -------------------------------------- |
> | Cómo cumple el proyecto **cada punto del enunciado** | notebook **§0** — mapa de cumplimiento |
> | El **dominio** y los 5 documentos                    | notebook **§1** · resumen abajo        |
> | La **arquitectura** del sistema                      | notebook **§2**                        |
> | La **justificación del system prompt**               | notebook **§10** · resumen abajo       |
> | El **agente LangGraph** y cómo funciona la memoria   | notebook **§11**                       |
> | Los **6 ejemplos documentados** (con sus respuestas) | notebook **§13**                       |
> | La **celda de chat interactiva**                     | notebook **§14**                       |
> | Las **decisiones técnicas** y su porqué              | notebook **§15**                       |
> | El **bonus de Streamlit**                            | notebook **§16** · `app.py`            |

---

## Inicio rápido

```bash
git clone <url-del-repo> && cd proyecto_final_rag

python -m venv .venv
source .venv/bin/activate          # Windows:  .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # y pegar dentro:  GOOGLE_API_KEY=tu_clave
```

**Notebook** (entregable principal):

```bash
jupyter notebook                   # abrir asistente_rag_gemini.ipynb y ejecutar en orden
```

**App web** (bonus):

```bash
streamlit run app.py
```

La primera ejecución descarga los 5 PDFs a `documents/` y genera la base vectorial en `chroma_db/`.
Las siguientes **reutilizan ambas cosas**: no se vuelve a descargar ni a indexar.

## Requisitos

- **Python 3.10+** (probado con 3.12).
- **API key de Google Gemini**.
  Se carga desde `.env`; **nunca se escribe en el código** (`.env` está en `.gitignore`).
- Conexión a internet **solo la primera vez**, para descargar los documentos.
- Dependencias en [`requirements.txt`](requirements.txt): `langchain`, `langchain-google-genai` (≥4.0),
  `langchain-chroma`, `langchain-community`, `langchain-text-splitters`, `langgraph`, `chromadb`,
  `pypdf`, `python-dotenv`, `requests`, `jupyter` · _(bonus)_ `streamlit`, `pysqlite3-binary`.

## El dominio

**Finanzas cuantitativas y mercados financieros**: un campo donde un LLM genérico _suena_
convincente pero falla en los detalles (VaR vs. Expected Shortfall, duración vs. convexidad). Es
justo donde el RAG aporta valor real: anclar cada respuesta en un texto académico concreto.

**5 PDFs, ~160 páginas** (el mínimo del enunciado eran 3 documentos / ~20 páginas). El
notebook **los descarga automáticamente** de sus fuentes oficiales (**§5**); no se redistribuyen
copias en este repositorio.

| Documento                      | Fuente                                              | Contenido                                      |
| ------------------------------ | --------------------------------------------------- | ---------------------------------------------- |
| `01_derivados_opciones.pdf`    | IARE — _Lecture Notes on Financial Derivatives_     | Forwards, futuros, call/put, valoración        |
| `02_teoria_carteras.pdf`       | Columbia, M. Haugh — _Mean-Variance & CAPM_         | Markowitz, frontera eficiente, ratio de Sharpe |
| `03_gestion_riesgo.pdf`        | Columbia, M. Haugh — _Quantitative Risk Management_ | VaR, Expected Shortfall, medidas coherentes    |
| `04_renta_fija.pdf`            | J. Wang — _Fixed Income Securities_ (MIT 15.401)    | Bonos, duración, convexidad                    |
| `05_procesos_estocasticos.pdf` | MIT OCW 15.433 — _Random Walk on Wall Street_       | Paseo aleatorio, mercados eficientes           |

Los documentos están **en inglés** y el agente responde **en español**: es deliberado — obliga al
modelo a comprender y traducir lo recuperado en vez de copiarlo, y demuestra que la búsqueda
semántica funciona a través del idioma.

## Justificación del system prompt

El prompt íntegro y el razonamiento completo están en el **§10 del notebook**. En corto, seis
decisiones:

1. **Rol y tono didáctico** — «QuantAsistente» debe **explicar**, no solo acertar: el usuario está aprendiendo.
2. **Fidelidad estricta al contexto** — responde _exclusivamente_ con lo recuperado. Es la salvaguarda principal **contra las alucinaciones**, el mayor riesgo de un asistente de dominio.
3. **Honestidad ante la ignorancia** — si el contexto no cubre la pregunta, lo dice. Un _«eso no está en mi base»_ es preferible a un dato falso pero convincente. → _demostrado en el ejemplo 5 (§13)_.
4. **Trazabilidad** — cita documento y página. Convierte la respuesta en **verificable**.
5. **Límite ético** — no predice precios ni recomienda comprar/vender; su contenido es **formativo y no es asesoramiento financiero**.
6. **Coherencia conversacional** — usa el historial para las preguntas de seguimiento.

## Bonus: interfaz Streamlit

`app.py` envuelve el mismo pipeline (embeddings → ChromaDB → grafo LangGraph con memoria) en una
interfaz de chat con **rastro de fuentes**: cada respuesta muestra el documento y la página de los
que sale, los fragmentos recuperados se pueden desplegar para verificarlos, y la **consulta
reformulada** se muestra cuando difiere de la pregunta original — la memoria del agente, visible.

**Desplegar en Streamlit Community Cloud**: conectar el repo en [share.streamlit.io](https://proyecto-rag-nihalhk.streamlit.app/)
con `app.py` como archivo principal, y añadir `GOOGLE_API_KEY` en los **secrets** .

Dos detalles **imprescindibles** para que el despliegue no se caiga:

- **`pysqlite3-binary`** en `requirements.txt` + la sustitución del módulo `sqlite3` al inicio de
  `app.py`: la imagen de Streamlit Cloud trae un `sqlite3` anterior al 3.35 que ChromaDB rechaza.
  **Sin ese parche la app ni siquiera arranca.**
- **`chroma_db/` se versiona a propósito** (por eso no está en `.gitignore`). El sistema de archivos
  de Streamlit Cloud es efímero: sin el índice en el repo, la app reindexaría ~300 fragmentos en
  **cada arranque en frío**, agotando la cuota gratuita de Gemini justo durante la demo.

## Estructura

```
proyecto_final_rag/
├── asistente_rag_gemini.ipynb   # ★ Entregable principal: código + documentación completa
├── app.py                       # Bonus: interfaz web Streamlit
├── requirements.txt
├── .env.example                 # Plantilla para la API key
├── .streamlit/config.toml       # Tema de la app
├── chroma_db/                   # Base vectorial (versionada: la necesita el despliegue)
├── documents/                   # Vacía: el notebook descarga aquí los 5 PDFs
└── README.md
```

## Límites conocidos

- **Memoria en RAM.** `MemorySaver` la pierde al reiniciar el kernel. Para persistirla entre
  sesiones, sustituirlo por `SqliteSaver` (**§17** del notebook).
- **Capa gratuita de Gemini.** Hay límite de peticiones por minuto. El proyecto lo gestiona con
  reintento automático (`con_reintento()`, **§8**), pero ante un `429` persistente conviene esperar.
- **El agente solo sabe lo que hay en sus 5 documentos.** No es un fallo: es el diseño.
- **Contenido formativo.** Nada de lo que responde constituye asesoramiento financiero.

---

Los 5 documentos son material docente de acceso público, propiedad de sus autores e instituciones
(MIT, Columbia University, IARE), usados aquí **exclusivamente con fines educativos**.
