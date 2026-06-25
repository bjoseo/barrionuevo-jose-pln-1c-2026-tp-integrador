# barrionuevo-jose-pln-1c-2026-tp-integrador

Repositorio TP PLN 1C-2026 para Entrega del Trabajo Integrador
Asistente RAG Local: Sostenibilidad e IA en América Latina
Proyecto Integrador — Laboratorio de PLN: Analítica, Textos y Cultura (2026) Autor original en notebook: Matías Barreto — IFTS24

RAG de IMPACTO ECONÓMICO DE LA INTELIGENCIA ARTIFICIAL EN AMÉRICA LATINA Juan Jung Raúl Katz Transformación tecnológica y rezago en materia de inversión y capacidades laborales

 Documentos: informe CENIA CEPAL 2025.pdf, Informe CEPAL 2025.pdf
 
Este repositorio contiene un pipeline completo de Retrieval-Augmented Generation (RAG) diseñado para ejecutarse de forma 100% local. El sistema permite consultar documentos técnicos e informes (como los de la CEPAL) sobre sostenibilidad, transición energética e Inteligencia Artificial en América Latina.

La aplicación detecta automáticamente el hardware disponible (CPU, GPU CUDA o Apple Silicon) para optimizar la inferencia del modelo de lenguaje, garantizando privacidad total y eficiencia en la ejecución.

Características Principales
Ejecución 100% Local: Privacidad garantizada al no depender de APIs externas (como OpenAI) para la inferencia del modelo de lenguaje.

Detección Automática de Hardware: Ajusta dinámicamente el modelo (gemma4:e2b o granite4:latest) y el contexto según los recursos del sistema operativo y la GPU.

Embeddings Multilingües: Utiliza paraphrase-multilingual-MiniLM-L12-v2 de Hugging Face para capturar la semántica de documentos en español con alta precisión.

Base de Datos Vectorial Persistente: Implementación de ChromaDB con cálculos de similitud coseno y técnicas avanzadas de recuperación (Threshold filtering y Metadata filtering).

Interfaz Gráfica Interactiva: Chat UI amigable construida con Gradio, que además cita las fuentes exactas y la página de donde extrae la información.

Stack Tecnológico
Orquestación: LangChain (LCEL)

LLM Local: Ollama

Embeddings: Sentence-Transformers (Hugging Face)

Vector Store: ChromaDB

Interfaz: Gradio

Procesamiento de Datos: PyPDF, LangChain Text Splitters

Guía de Instalación y Uso

1. Requisitos Previos
   Python 3.10+ instalado en tu sistema.

Ollama instalado y ejecutándose. Podés descargarlo desde ollama.com.

Descargar el modelo base a utilizar en Ollama. Abrí una terminal y ejecutá:

ollama pull gemma4:e2b

2. Clonar el Repositorio
   Si estás trabajando en equipo, recordá mantener tu rama sincronizada realizando un fetch y pull periódicamente.

git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

3. Entorno Virtual e Instalación de Dependencias
   Se recomienda utilizar un entorno virtual para aislar las dependencias del proyecto.

# Crear entorno virtual

python -m venv .venv

# Activar entorno (Windows)

.venv\Scripts\activate

# Activar entorno (Linux/Mac)

source .venv/bin/activate

# Instalar dependencias

pip install -r requirements.txt

(Nota: Asegurate de tener un archivo requirements.txt generado, o bien ejecuta las instalaciones mencionadas en el notebook).

4. Preparación de Datos
   Colocá tu corpus de documentos en formato JSON en la ruta: data/corpus_limpio.json.

El sistema se encargará automáticamente de leer el archivo, generar los chunks (fragmentos de 1000 caracteres con 200 de solapamiento) y crear la base de datos vectorial en la carpeta chroma_db.

5. Ejecución
   Asegurate de que el servidor de Ollama esté corriendo en segundo plano:

ollama serve

Ejecutá el notebook pipeline_rag.ipynb paso a paso, o bien el script de Python si lo exportaste.

Al finalizar, Gradio levantará un servidor web local (por defecto en http://127.0.0.1:7861). Hacé clic en el enlace de la terminal para abrir la interfaz en tu navegador y comenzar a chatear con los documentos.

🧠 Estructura del Pipeline RAG
1.- Ingesta: Carga del corpus_limpio.json.

2.- Chunking: División estratégica del texto usando RecursiveCharacterTextSplitter.

3.- Indexación: Los fragmentos se vectorizan y almacenan en ChromaDB.

4.- Recuperación (Retrieval): Búsqueda de los K fragmentos más relevantes basándose en similitud coseno, con posibilidad de aplicar umbrales de relevancia.

5.- Generación: Se construye un prompt inyectando el contexto recuperado y la pregunta del usuario. Ollama genera la respuesta final basándose únicamente en esa información.
