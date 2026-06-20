import os
import json
from pypdf import PdfReader

def extraer_texto_de_un_pdf(ruta_pdf):
    """
    Lee un archivo PDF específico y extrae el texto de cada página
    retornando una lista de diccionarios con su metadata.
    """
    reader = PdfReader(ruta_pdf)
    
    # Intentamos extraer metadata interna del PDF
    meta = reader.metadata
    autor_pdf = meta.author if meta and meta.author else "Organismo Internacional"
    titulo_pdf = meta.title if meta and meta.title else os.path.basename(ruta_pdf)
    
    # Intentamos deducir el año o la fuente según el nombre del archivo
    nombre_archivo_lower = os.path.basename(ruta_pdf).lower()
    fuente = "CEPAL" if "cepal" in nombre_archivo_lower else ("BID" if "bid" in nombre_archivo_lower else "Desconocida")
    
    # Intenta buscar un año de 4 dígitos en el nombre del archivo (ej: 2025, 2026)
    anio_detectado = "2026"
    for palabra in nombre_archivo_lower.replace("_", " ").replace("-", " ").split():
        if palabra.isdigit() and len(palabra) == 4:
            anio_detectado = palabra
            break

    paginas_archivo = []

    for numero_pagina, pagina in enumerate(reader.pages, start=1):
        texto_crudo = pagina.extract_text()
        if not texto_crudo:
            continue  # Saltea páginas vacías o imágenes puras
            
        # Limpieza inicial básica de espacios para que no se rompa la estructura
        texto_limpio = " ".join(texto_crudo.split())
        
        datos_pagina = {
            "contenido": texto_limpio,
            "metadata": {
                "fuente": fuente,
                "archivo_origen": os.path.basename(ruta_pdf),
                "titulo": titulo_pdf,
                "autor": autor_pdf,
                "anio": anio_detectado,
                "pagina": numero_pagina
            }
        }
        paginas_archivo.append(datos_pagina)
        
    return paginas_archivo

if __name__ == "__main__":
    # 1. Ubicar la carpeta del proyecto de forma dinámica
    ruta_del_script = os.path.dirname(os.path.abspath(__file__))
    carpeta_data = os.path.join(ruta_del_script, "data")
    os.makedirs(carpeta_data, exist_ok=True)
    
    # 2. Escanear la carpeta buscando todos los PDFs
    archivos_en_carpeta = os.listdir(carpeta_data)
    pdfs_disponibles = [f for f in archivos_en_carpeta if f.lower().endswith('.pdf')]
    
    print("=== Pipeline de Ingesta Masiva ===")
    print(f"Buscando PDFs en: {carpeta_data}\n")
    
    if not pdfs_disponibles:
        print("[ALERTA] No hay archivos PDF en la carpeta 'data'.")
        print("Copiá los informes allí y volvé a ejecutar el script.")
    else:
        print(f"[INFO] Se encontraron {len(pdfs_disponibles)} archivos para procesar.")
        
        # Lista gigante donde consolidaremos todas las páginas de todos los libros
        todo_el_corpus = []
        
        # 3. Iterar sobre cada PDF encontrado
        for nombre_pdf in pdfs_disponibles:
            ruta_completa_pdf = os.path.join(carpeta_data, nombre_pdf)
            try:
                paginas_procesadas = extraer_texto_de_un_pdf(ruta_completa_pdf)
                todo_el_corpus.extend(paginas_procesadas)
                print(f" -> [OK] '{nombre_pdf}' procesado con éxito ({len(paginas_procesadas)} páginas).")
            except Exception as e:
                print(f" -> [ERROR] No se pudo procesar el archivo '{nombre_pdf}'. Motivo: {e}")
        
        # 4. Guardar todo el corpus consolidado en el JSON definitivo
        ruta_salida_consolidada = os.path.join(carpeta_data, "corpus_sostenibilidad.json")
        
        with open(ruta_salida_consolidada, "w", encoding="utf-8") as f:
            json.dump(todo_el_corpus, f, ensure_ascii=False, indent=4)
            
        print("\n==================================================")
        print("[PROCESO COMPLETADO] ¡Corpus consolidado creado!")
        print(f"Archivo generado: {ruta_salida_consolidada}")
        print(f"Total de páginas integradas en el sistema: {len(todo_el_corpus)}")
        print("==================================================")