import gradio as gr
import os
import json
import re
import shutil
from pypdf import PdfReader

# --- IMPORTACIONES DE LANGCHAIN & HUGGING FACE ---
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN BASE DE CARPETAS ---
CARPETA_DATOS = "data"
CARPETA_CHROMA = "chroma_db"

# Forzamos la creación de ambas carpetas al iniciar
os.makedirs(CARPETA_DATOS, exist_ok=True)
os.makedirs(CARPETA_CHROMA, exist_ok=True)

# --- 1. CARGA DE MODELOS EN MEMORIA (Vía API de HF) ---
print("[INFO] Cargando el modelo de embeddings...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

print("[INFO] Conectando con el LLM en Hugging Face Serverless API...")
# Utiliza automáticamente el HF_TOKEN configurado en los Secrets del Space
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-1.1-2b-it", # Modelo ágil y excelente para RAG
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False,
)
print("[OK] Modelos inicializados exitosamente.")

# --- VARIABLES GLOBALES DEL SISTEMA RAG ---
vector_store = None
retriever_activo = None
pipeline_rag = None
corpus_activo_nombre = None

# --- 2. FUNCIONES DE GESTIÓN DE ARCHIVOS ---

def listar_corpus_disponibles():
    archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith('.json')]
    return archivos if archivos else ["No hay corpus creados aún"]

def extraer_y_limpiar_pdf(ruta_pdf, nombre_original):
    documentos = []
    try:
        reader = PdfReader(ruta_pdf)
        for i, pagina in enumerate(reader.pages):
            texto = pagina.extract_text()
            if texto:
                texto_limpio = re.sub(r'\s+', ' ', texto).strip()
                if texto_limpio:
                    doc = {
                        "contenido": texto_limpio,
                        "metadata": {"fuente": nombre_original, "pagina": i + 1}
                    }
                    documentos.append(doc)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el PDF {nombre_original}: {e}")
    return documentos

def procesar_pdf(archivos_pdf, nombre_nuevo_corpus, corpus_existente, accion):
    if not archivos_pdf:
        return "⚠️ Por favor, subí al menos un PDF.", gr.update(), gr.update(), gr.update()
    
    if accion == "Crear nuevo corpus":
        if not nombre_nuevo_corpus:
            return "⚠️ Ingresá un nombre para el nuevo corpus.", gr.update(), gr.update(), gr.update()
        nombre_archivo = f"{nombre_nuevo_corpus.strip().replace(' ', '_')}.json"
        ruta_json = os.path.join(CARPETA_DATOS, nombre_archivo)
        corpus_data = []
    else:
        if not corpus_existente or corpus_existente == "No hay corpus creados aún":
            return "⚠️ Seleccioná un corpus válido existente.", gr.update(), gr.update(), gr.update()
        ruta_json = os.path.join(CARPETA_DATOS, corpus_existente)
        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
        except Exception as e:
            return f"⚠️ Error al leer el corpus existente: {e}", gr.update(), gr.update(), gr.update()

    for pdf in archivos_pdf:
        ruta_temp = pdf.name
        nombre_original = os.path.basename(ruta_temp.replace('\\', '/')) 
        nuevos_documentos = extraer_y_limpiar_pdf(ruta_temp, nombre_original)
        corpus_data.extend(nuevos_documentos)
    
    try:
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        nombre_sin_ext = os.path.basename(ruta_json).replace(".json", "")
        ruta_vector_cache = os.path.join(CARPETA_CHROMA, nombre_sin_ext)
        if os.path.exists(ruta_vector_cache):
            shutil.rmtree(ruta_vector_cache)
            
    except Exception as e:
        return f"⚠️ Error al guardar el JSON: {e}", gr.update(), gr.update(), gr.update()
    
    lista_actualizada = listar_corpus_disponibles()
    return "✅ Corpus guardado. Los vectores se actualizarán al cargarlo en la pestaña RAG.", gr.update(choices=lista_actualizada), gr.update(choices=lista_actualizada), gr.update(choices=lista_actualizada)

def eliminar_corpus(corpus_seleccionado):
    if not corpus_seleccionado or corpus_seleccionado == "No hay corpus creados aún":
        return "⚠️ Seleccioná un corpus para eliminar.", gr.update(), gr.update(), gr.update()
    
    ruta_json = os.path.join(CARPETA_DATOS, corpus_seleccionado)
    nombre_sin_ext = corpus_seleccionado.replace(".json", "")
    ruta_vector_db = os.path.join(CARPETA_CHROMA, nombre_sin_ext)

    try:
        if os.path.exists(ruta_json):
            os.remove(ruta_json)
        if os.path.exists(ruta_vector_db):
            shutil.rmtree(ruta_vector_db)
            
        lista_actualizada = listar_corpus_disponibles()
        return f"🗑️ Se eliminó por completo el corpus '{corpus_seleccionado}'.", gr.update(choices=lista_actualizada, value=None), gr.update(choices=lista_actualizada, value=None), gr.update(choices=lista_actualizada, value=None)
    except Exception as e:
        return f"⚠️ Error al eliminar: {e}", gr.update(), gr.update(), gr.update()

# --- 3. LÓGICA DINÁMICA DEL PIPELINE RAG ---

def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def cargar_corpus_en_rag(corpus_seleccionado):
    global vector_store, retriever_activo, pipeline_rag, corpus_activo_nombre

    if not corpus_seleccionado or corpus_seleccionado == "No hay corpus creados aún":
        return "⚠️ Seleccioná un corpus válido primero."

    ruta_json = os.path.join(CARPETA_DATOS, corpus_seleccionado)
    nombre_sin_ext = corpus_seleccionado.replace(".json", "")
    ruta_vector_db = os.path.join(CARPETA_CHROMA, nombre_sin_ext)

    TEMPLATE_PROMPT = """Eres un analista de datos experto. Tu única tarea es responder la Pregunta del usuario utilizando ÚNICAMENTE la información provista en los Documentos de contexto.

INSTRUCCIONES CLAVE:
1. Lee TODOS los documentos de contexto detenidamente antes de responder.
2. Si encuentras la respuesta, redáctala de forma clara y directa.
3. Si la respuesta definitiva no está, pero hay información relacionada, resúmela y aclara que es un dato parcial.
4. Solo di "No encontré la información" si realmente no hay nada útil en el texto.

Documentos de contexto:
{context}

Pregunta del usuario: {question}

Respuesta analítica:"""

    prompt = PromptTemplate(template=TEMPLATE_PROMPT, input_variables=["context", "question"])

    def construir_pipeline():
        global pipeline_rag, retriever_activo, corpus_activo_nombre
        
        # Implementación con MMR: Busca 20 fragmentos, pero se queda con los 5 mejores y más diversos
        retriever_activo = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5, 
                "fetch_k": 20
            }
        )
        
        pipeline_rag = (
            {"context": retriever_activo | formatear_documentos, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        corpus_activo_nombre = nombre_sin_ext
    if os.path.exists(ruta_vector_db) and len(os.listdir(ruta_vector_db)) > 0:
        try:
            vector_store = Chroma(persist_directory=ruta_vector_db, embedding_function=embeddings_model)
            construir_pipeline()
            return f"🚀 ¡Corpus '{nombre_sin_ext}' montado AL INSTANTE desde caché!"
        except Exception as e:
            return f"⚠️ Error al cargar almacenamiento persistente: {e}"

    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos_corpus = json.load(f)

        if not datos_corpus:
            return "⚠️ El archivo JSON seleccionado está vacío."

        documentos_langchain = [Document(page_content=doc["contenido"], metadata=doc["metadata"]) for doc in datos_corpus]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        documentos_fragmentados = splitter.split_documents(documentos_langchain)

        vector_store = Chroma.from_documents(
            documents=documentos_fragmentados,
            embedding=embeddings_model,
            persist_directory=ruta_vector_db,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        construir_pipeline()
        return f"✅ Indexación completada con éxito. Se procesaron {len(documentos_fragmentados)} fragmentos."

    except Exception as e:
        return f"⚠️ Error crítico al indexar el corpus: {e}"

# --- 4. FUNCIÓN EJECUTORA DEL CHAT ---

def responder_rag_interactivo(mensaje, historial):
    global pipeline_rag, retriever_activo, corpus_activo_nombre

    if pipeline_rag is None or retriever_activo is None:
        return "⚠️ El sistema RAG no está activo. Por favor, ve a la pestaña '⚙️ Configuración del RAG' y carga un corpus."

    try:
        docs_fuentes = retriever_activo.invoke(mensaje)
        fuentes_usadas = f"\n\n--- 📄 FUENTES UTILIZADAS ({corpus_activo_nombre}) ---"
        if docs_fuentes:
            for i, doc in enumerate(docs_fuentes, 1):
                fuentes_usadas += f"\n• [Ref {i}] Archivo: {doc.metadata.get('fuente')} - Pág. {doc.metadata.get('pagina')}"
        else:
            fuentes_usadas += "\n• No se encontraron fragmentos con suficiente similitud semántica."

        respuesta_llm = pipeline_rag.invoke(mensaje)
        return f"{respuesta_llm}{fuentes_usadas}"
        
    except Exception as e:
        return f"❌ Error al procesar la consulta con el modelo: {str(e)}"

# --- 5. INTERFAZ GRÁFICA DE GRADIO (UI) ---

css_estilos = """
#caja-chat textarea {
    background-color: #F0F8FF !important;
    border: 2px solid #4A90E2 !important;
    font-size: 16px !important;
    padding: 15px !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css_estilos) as demo:
    gr.Markdown("# 🤖 Plataforma RAG Avanzada (Multicorpus)")
    gr.Markdown("Gestioná múltiples repositorios de conocimiento e indexá PDFs bajo demanda.")
    
    lista_inicial = listar_corpus_disponibles()
    
    with gr.Tabs():
        # PESTAÑA 1
        with gr.Tab("📁 Gestión de Corpus"):
            with gr.Row():
                with gr.Column():
                    archivos_pdf = gr.File(label="Subir PDFs", file_count="multiple", file_types=[".pdf"])
                    accion_radio = gr.Radio(["Crear nuevo corpus", "Agregar a corpus existente"], label="Acción", value="Crear nuevo corpus")
                with gr.Column():
                    nombre_nuevo = gr.Textbox(label="Nombre nuevo corpus", visible=True)
                    dropdown_existente_gestion = gr.Dropdown(choices=lista_inicial, label="Seleccionar corpus", visible=False)
                    btn_procesar = gr.Button("⚙️ Procesar y Guardar JSON", variant="primary")
            
            mensaje_gestion = gr.Textbox(label="Estado de carga", interactive=False)
            gr.Markdown("---")
            with gr.Row():
                dropdown_eliminar = gr.Dropdown(choices=lista_inicial, label="Seleccionar corpus a eliminar", scale=3)
                btn_eliminar = gr.Button("Eliminar Corpus Definitivamente", variant="stop", scale=1)
            mensaje_eliminar = gr.Textbox(label="Estado de eliminación", interactive=False)
            
            def actualizar_visibilidad(accion):
                if accion == "Crear nuevo corpus": return gr.update(visible=True), gr.update(visible=False)
                else: return gr.update(visible=False), gr.update(visible=True)
            accion_radio.change(fn=actualizar_visibilidad, inputs=accion_radio, outputs=[nombre_nuevo, dropdown_existente_gestion])
            
        # PESTAÑA 2
        with gr.Tab("⚙️ Configuración del RAG"):
            with gr.Row():
                dropdown_cargar_rag = gr.Dropdown(choices=lista_inicial, label="Corpus disponible")
                btn_cargar_rag = gr.Button("🚀 Cargar al Sistema RAG", variant="primary")
            mensaje_carga = gr.Textbox(label="Estado del Sistema RAG", interactive=False)
            btn_cargar_rag.click(fn=cargar_corpus_en_rag, inputs=dropdown_cargar_rag, outputs=mensaje_carga)

        # PESTAÑA 3
        with gr.Tab("💬 Chat"):
            gr.Markdown("### 💬 Consulta Inteligente de Documentos")
            chatbot = gr.ChatInterface(
                fn=responder_rag_interactivo,
                textbox=gr.Textbox(
                    placeholder="✍️ Escribí tu consulta aquí y presioná Enter...", 
                    container=True, 
                    scale=7,
                    elem_id="caja-chat"
                ),
            )

        # Eventos
        btn_procesar.click(
            fn=procesar_pdf, 
            inputs=[archivos_pdf, nombre_nuevo, dropdown_existente_gestion, accion_radio], 
            outputs=[mensaje_gestion, dropdown_existente_gestion, dropdown_eliminar, dropdown_cargar_rag]
        )
        btn_eliminar.click(
            fn=eliminar_corpus,
            inputs=dropdown_eliminar,
            outputs=[mensaje_eliminar, dropdown_existente_gestion, dropdown_eliminar, dropdown_cargar_rag]
        )

# Para HF Spaces, no se usa share=True
if __name__ == "__main__":
    demo.launch()