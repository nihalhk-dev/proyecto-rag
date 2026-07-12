"""
app.py — Interfaz web (BONUS) del Asistente Experto con Gemini, RAG y Agentes.

Reutiliza el mismo pipeline del notebook (Gemini Embeddings -> ChromaDB -> grafo
LangGraph con memoria) y añade:
  · Reintento automático ante el límite de cuota de la capa gratuita (error 429).
  · Indexación por lotes si la base vectorial aún no existe.
  · Rastro de fuentes: cada respuesta muestra de qué documento y página proviene.

Ejecutar en local:
    streamlit run app.py

Despliegue en Streamlit Community Cloud: sube el repositorio y añade GOOGLE_API_KEY
en los "secrets" de la app (nunca en el código).
"""

# --- Parche de sqlite3 para Streamlit Cloud -----------------------------------
# ChromaDB exige sqlite3 >= 3.35, pero la imagen de Streamlit Community Cloud trae
# una versión más antigua y la app falla al arrancar con:
#   "RuntimeError: Your system has an unsupported version of sqlite3".
# La solución oficial es instalar 'pysqlite3-binary' (ya está en requirements.txt)
# y sustituir el módulo sqlite3 ANTES de importar chromadb. En local no hace nada.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # en local se usa el sqlite3 del sistema, que ya es lo bastante nuevo
# ------------------------------------------------------------------------------

import os
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ============================ Configuración ============================
CARPETA_DOCS = Path("documents")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "finanzas_cuantitativas"

MODELO_LLM = "gemini-2.5-flash"
MODELO_EMBEDDING = "gemini-embedding-001"

CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1200, 200, 4
TAMANO_LOTE, PAUSA_ENTRE_LOTES = 20, 3

# Documentos reales de la base de conocimiento (mismas fuentes que el notebook).
DOCUMENTOS_REALES = {
    "01_derivados_opciones.pdf": (
        "https://www.iare.ac.in/sites/default/files/lecture_notes/IARE_FD_NOTES.pdf",
        "Derivados y opciones",
        "IARE · Lecture Notes on Financial Derivatives",
    ),
    "02_teoria_carteras.pdf": (
        "https://www.columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf",
        "Teoría de carteras y CAPM",
        "Columbia University · M. Haugh",
    ),
    "03_gestion_riesgo.pdf": (
        "http://www.columbia.edu/~mh2078/RiskMeasures.pdf",
        "Gestión del riesgo (VaR)",
        "Columbia University · M. Haugh",
    ),
    "04_renta_fija.pdf": (
        "https://www.its.caltech.edu/~rosentha/courses/BEM103/Readings/JWCh03.pdf",
        "Renta fija y duración",
        "MIT 15.401 · J. Wang",
    ),
    "05_procesos_estocasticos.pdf": (
        "https://ocw.mit.edu/courses/15-433-investments-spring-2003/c5845cd981c2e63f7ff303c92c7d41be_154332random_walk.pdf",
        "Procesos estocásticos",
        "MIT OCW 15.433",
    ),
}

PREGUNTAS_SUGERIDAS = [
    "¿Qué es el ratio de Sharpe y cómo se interpreta?",
    "¿Qué diferencia hay entre una opción call y una put?",
    "¿Por qué se dice que el VaR no es una medida coherente de riesgo?",
    "¿Cómo afecta la duración al precio de un bono?",
]

SYSTEM_PROMPT = (
    "Eres «QuantAsistente», un asistente experto en finanzas cuantitativas y mercados "
    "financieros. Tu misión es responder preguntas basándote EXCLUSIVAMENTE en el "
    "CONTEXTO recuperado de una base de conocimiento especializada (apuntes académicos "
    "de MIT, Columbia University y otras instituciones).\n\n"
    "REGLAS DE COMPORTAMIENTO:\n"
    "1. Responde siempre en español, con un tono claro, riguroso y didáctico, aunque el "
    "material fuente esté en inglés.\n"
    "2. Usa ÚNICAMENTE la información del CONTEXTO. No inventes datos, cifras ni fórmulas.\n"
    "3. Si el CONTEXTO no contiene la información necesaria, dilo con honestidad en lugar "
    "de improvisar.\n"
    "4. Cuando sea útil, menciona de qué documento procede la información.\n"
    "5. No haces predicciones de precios ni recomendaciones de compra/venta. Tus "
    "explicaciones son formativas y NO constituyen asesoramiento financiero.\n"
    "6. Ten en cuenta el historial de la conversación para mantener la coherencia."
)

# ------------- Clave de API: .env (local) o secrets (Streamlit Cloud) -------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
try:
    API_KEY = API_KEY or st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY


# ==================== Resiliencia frente al límite de cuota ====================
def _espera_sugerida(error: Exception, por_defecto: float = 60.0) -> float:
    """Lee el 'Please retry in N s' que devuelve Gemini al agotar la cuota."""
    m = re.search(r"retry in ([\d.]+)s", str(error))
    return float(m.group(1)) + 2 if m else por_defecto


def con_reintento(func, *args, max_intentos: int = 5, **kwargs):
    """Ejecuta func(); si Gemini responde 429 (cuota agotada), espera y reintenta.

    La capa gratuita limita las peticiones por minuto, así que sin esto la app se
    caería a mitad de una conversación.
    """
    for intento in range(1, max_intentos + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            agotada = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if agotada and intento < max_intentos:
                time.sleep(_espera_sugerida(e))
            else:
                raise


# ============================ Base de conocimiento ============================
def descargar_documentos():
    """Descarga los PDFs a documents/ si aún no están en disco."""
    CARPETA_DOCS.mkdir(exist_ok=True)
    for nombre, (url, _, _) in DOCUMENTOS_REALES.items():
        destino = CARPETA_DOCS / nombre
        if destino.exists():
            continue
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            destino.write_bytes(resp.content)
        except Exception as e:
            st.warning(f"No se pudo descargar {nombre}: {e}")


def cargar_y_trocear():
    """Carga los documentos de documents/ y los parte en fragmentos."""
    documentos = []
    for ruta in sorted(CARPETA_DOCS.rglob("*")):
        if ruta.is_dir():
            continue
        ext = ruta.suffix.lower()
        if ext == ".pdf":
            documentos.extend(PyPDFLoader(str(ruta)).load())
        elif ext in {".md", ".txt"}:
            documentos.extend(TextLoader(str(ruta), encoding="utf-8").load())
        elif ext == ".csv":
            documentos.extend(CSVLoader(str(ruta)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


@st.cache_resource(show_spinner=False)
def construir_agente():
    """Prepara modelos, base vectorial y grafo. Se ejecuta una sola vez por sesión."""
    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
    llm = ChatGoogleGenerativeAI(model=MODELO_LLM, temperature=0.2)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Solo indexamos si la colección está vacía (p. ej. en un despliegue nuevo).
    if vectorstore._collection.count() == 0:
        with st.status("Preparando la base de conocimiento…", expanded=True) as estado:
            estado.write("Descargando los documentos académicos…")
            descargar_documentos()

            estado.write("Segmentando el texto en fragmentos…")
            fragmentos = cargar_y_trocear()

            barra = st.progress(0.0, text=f"Indexando 0 / {len(fragmentos)} fragmentos")
            for i in range(0, len(fragmentos), TAMANO_LOTE):
                lote = fragmentos[i: i + TAMANO_LOTE]
                con_reintento(vectorstore.add_documents, lote)
                hechos = min(i + TAMANO_LOTE, len(fragmentos))
                barra.progress(hechos / len(fragmentos),
                               text=f"Indexando {hechos} / {len(fragmentos)} fragmentos")
                time.sleep(PAUSA_ENTRE_LOTES)
            estado.update(label="Base de conocimiento lista", state="complete", expanded=False)

    # ---------------------------- Grafo LangGraph ----------------------------
    class EstadoChat(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        context: str
        consulta: str
        fuentes: list

    def reformular(pregunta: str, historial: list) -> str:
        """Convierte una pregunta de seguimiento en una consulta autónoma y buscable."""
        texto = "\n".join(
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
            for m in historial[-6:]
        )
        instruccion = (
            "Dada la conversación previa y una nueva pregunta, reescribe la nueva pregunta "
            "como una consulta de búsqueda AUTÓNOMA y completa, entendible por sí sola. Si "
            "ya es autónoma, devuélvela tal cual. Responde SOLO con la consulta.\n\n"
            f"CONVERSACIÓN:\n{texto}\n\nNUEVA PREGUNTA: {pregunta}\n\nCONSULTA AUTÓNOMA:"
        )
        return con_reintento(llm.invoke, instruccion).content.strip()

    def recuperar(estado: EstadoChat) -> dict:
        mensajes = estado["messages"]
        pregunta = mensajes[-1].content
        historial = mensajes[:-1]

        consulta = reformular(pregunta, historial) if historial else pregunta
        docs = con_reintento(vectorstore.similarity_search, consulta, k=TOP_K)

        contexto = "\n\n---\n\n".join(
            f"[Fuente: {Path(d.metadata.get('source', '?')).name}, "
            f"pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
        fuentes = [
            {
                "archivo": Path(d.metadata.get("source", "?")).name,
                "pagina": d.metadata.get("page", "?"),
                "extracto": d.page_content[:320].replace("\n", " ").strip(),
            }
            for d in docs
        ]
        return {"context": contexto, "consulta": consulta, "fuentes": fuentes}

    def generar(estado: EstadoChat) -> dict:
        pregunta = estado["messages"][-1].content
        historial = estado["messages"][:-1]
        prompt_turno = (
            "Responde a la PREGUNTA usando exclusivamente el siguiente CONTEXTO.\n\n"
            f"CONTEXTO:\n{estado['context']}\n\nPREGUNTA: {pregunta}"
        )
        mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + historial + [
            HumanMessage(content=prompt_turno)
        ]
        return {"messages": [con_reintento(llm.invoke, mensajes)]}

    grafo = StateGraph(EstadoChat)
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("generar", generar)
    grafo.add_edge(START, "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    agente = grafo.compile(checkpointer=MemorySaver())
    return agente, vectorstore._collection.count()


# ================================ Estilos ================================
ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,500;1,300&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --tinta:        #131A2B;
  --tinta-alta:   #1B2438;
  --regla:        #2E3A57;
  --pergamino:    #E9E3D6;
  --pergamino-2:  #96A0B4;
  --laton:        #C9A227;
}

.stApp { background: var(--tinta); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 820px; }

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* ---------- Cabecera ---------- */
.cabecera { margin-bottom: 1.6rem; }
.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .68rem; letter-spacing: .22em; text-transform: uppercase;
  color: var(--laton); margin-bottom: .5rem;
}
.titulo {
  font-family: 'Spectral', Georgia, serif;
  font-weight: 300; font-size: 2.9rem; line-height: 1.05;
  color: var(--pergamino); letter-spacing: -.015em; margin: 0;
}
.subtitulo {
  font-size: .93rem; color: var(--pergamino-2);
  margin-top: .55rem; max-width: 52ch;
}
.regla-laton {
  height: 1px; margin: 1.3rem 0 0;
  background: linear-gradient(90deg, var(--laton) 0%, var(--regla) 45%, transparent 100%);
}

/* ---------- Rastro de fuentes (elemento firma) ---------- */
.rastro { margin: .55rem 0 .2rem; padding-left: .85rem; border-left: 2px solid var(--laton); }
.rastro-titulo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .66rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--laton); margin-bottom: .45rem;
}
.chip {
  display: inline-block; margin: 0 .35rem .35rem 0; padding: .2rem .55rem;
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
  color: var(--pergamino); background: var(--tinta-alta);
  border: 1px solid var(--regla); border-radius: 3px;
}
.chip .pag { color: var(--pergamino-2); }
.reformulada {
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
  color: var(--pergamino-2); margin-top: .4rem;
}
.reformulada b { color: var(--laton); font-weight: 500; }
.extracto {
  font-size: .82rem; color: var(--pergamino-2); line-height: 1.55;
  border-left: 1px solid var(--regla); padding-left: .8rem; margin-bottom: .9rem;
}
.extracto .ref {
  display: block; font-family: 'IBM Plex Mono', monospace;
  font-size: .68rem; color: var(--laton); margin-bottom: .25rem;
}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"] { background: var(--tinta-alta); border-right: 1px solid var(--regla); }
.lat-titulo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .66rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--laton); margin: .2rem 0 .7rem;
}
.doc { margin-bottom: .75rem; }
.doc-nombre { font-size: .85rem; color: var(--pergamino); }
.doc-fuente {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .68rem; color: var(--pergamino-2); margin-top: .1rem;
}
.metrica {
  font-family: 'Spectral', serif; font-size: 2rem; font-weight: 300;
  color: var(--laton); line-height: 1;
}
.metrica-pie {
  font-family: 'IBM Plex Mono', monospace; font-size: .66rem;
  letter-spacing: .12em; text-transform: uppercase; color: var(--pergamino-2);
}
.aviso {
  font-size: .74rem; color: var(--pergamino-2); line-height: 1.5;
  border-top: 1px solid var(--regla); padding-top: .8rem; margin-top: .3rem;
}

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""

# ================================ Interfaz ================================
st.set_page_config(page_title="QuantAsistente", page_icon="§", layout="centered")
st.markdown(ESTILOS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="cabecera">
      <div class="eyebrow">RAG · Gemini · ChromaDB · LangGraph</div>
      <h1 class="titulo">QuantAsistente</h1>
      <div class="subtitulo">
        Responde sobre finanzas cuantitativas leyendo apuntes de MIT y Columbia.
        Cada respuesta indica el documento y la página de los que sale.
      </div>
      <div class="regla-laton"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not API_KEY:
    st.error(
        "Falta la clave de Gemini. Añade GOOGLE_API_KEY a un archivo .env (en local) "
        "o a los secrets de la app (en Streamlit Cloud)."
    )
    st.stop()

agente, n_fragmentos = construir_agente()

# ---- Estado de la sesión ----
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("historial", [])
st.session_state.setdefault("pendiente", None)


def formatear_rastro(fuentes: list, consulta: str, pregunta: str) -> str:
    """Construye el rastro de fuentes que acompaña a cada respuesta."""
    vistas, chips = set(), []
    for f in fuentes:
        clave = (f["archivo"], f["pagina"])
        if clave in vistas:
            continue
        vistas.add(clave)
        titulo = DOCUMENTOS_REALES.get(f["archivo"], (None, f["archivo"], None))[1]
        chips.append(f'<span class="chip">{titulo} <span class="pag">· p. {f["pagina"]}</span></span>')

    html = ['<div class="rastro"><div class="rastro-titulo">Fuentes consultadas</div>']
    html.append("".join(chips))
    if consulta and consulta.strip().lower() != pregunta.strip().lower():
        html.append(f'<div class="reformulada"><b>Búsqueda reformulada →</b> {consulta}</div>')
    html.append("</div>")
    return "".join(html)


# ---- Barra lateral ----
with st.sidebar:
    st.markdown('<div class="lat-titulo">Base de conocimiento</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metrica">{n_fragmentos}</div>'
        f'<div class="metrica-pie">fragmentos indexados</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)

    for _, (_, titulo, fuente) in DOCUMENTOS_REALES.items():
        st.markdown(
            f'<div class="doc"><div class="doc-nombre">{titulo}</div>'
            f'<div class="doc-fuente">{fuente}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="lat-titulo" style="margin-top:1.4rem">Prueba a preguntar</div>',
                unsafe_allow_html=True)
    for sugerencia in PREGUNTAS_SUGERIDAS:
        if st.button(sugerencia, use_container_width=True, key=f"sug_{sugerencia[:18]}"):
            st.session_state.pendiente = sugerencia

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    if st.button("Empezar de nuevo", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.historial = []
        st.rerun()

    st.markdown(
        '<div class="aviso">Contenido formativo. No es asesoramiento financiero.</div>',
        unsafe_allow_html=True,
    )

# ---- Conversación ----
for turno in st.session_state.historial:
    with st.chat_message(turno["rol"]):
        st.write(turno["texto"])
        if turno.get("rastro"):
            st.markdown(turno["rastro"], unsafe_allow_html=True)
            with st.expander("Ver los fragmentos recuperados"):
                for f in turno["fuentes"]:
                    st.markdown(
                        f'<div class="extracto"><span class="ref">{f["archivo"]} · pág. {f["pagina"]}'
                        f'</span>{f["extracto"]}…</div>',
                        unsafe_allow_html=True,
                    )

pregunta = st.chat_input("Pregunta sobre finanzas cuantitativas…") or st.session_state.pendiente
st.session_state.pendiente = None

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.historial.append({"rol": "user", "texto": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los apuntes…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            resultado = agente.invoke(
                {"messages": [HumanMessage(content=pregunta)]}, config=config
            )
            respuesta = resultado["messages"][-1].content
            fuentes = resultado.get("fuentes", [])
            consulta = resultado.get("consulta", "")

        st.write(respuesta)
        rastro = formatear_rastro(fuentes, consulta, pregunta)
        st.markdown(rastro, unsafe_allow_html=True)
        with st.expander("Ver los fragmentos recuperados"):
            for f in fuentes:
                st.markdown(
                    f'<div class="extracto"><span class="ref">{f["archivo"]} · pág. {f["pagina"]}'
                    f'</span>{f["extracto"]}…</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.historial.append(
        {"rol": "assistant", "texto": respuesta, "rastro": rastro, "fuentes": fuentes}
    )
