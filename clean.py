import os
import json
import re

def limpiar_texto_avanzado(texto):
    """
    Limpia el texto conservando acentos, eñes y eliminando encabezados específicos.
    """
    # 1. ELIMINAR ENCABEZADO DINÁMICO
    # REEMPLAZÁ ESTA FRASE por el encabezado exacto que ves en tu JSON.
    # El '\d+' al final le dice a Python que borre la frase seguida de cualquier número.
    patron_encabezado = r"CEPAL Impacto económico de la inteligencia artificial en América Latina...*?\d+"
    texto = re.sub(patron_encabezado, "", texto)
    patron_encabezado = r"ILIA 2025.*?\d+"
    texto = re.sub(patron_encabezado, "", texto)
    
    # [Alternativa alternativa por longitud]: Si el encabezado varía un poco pero siempre 
    # empieza igual, podemos usar un patrón que borre la frase fija y hasta 5 caracteres más:
    # texto = re.sub(r"Tu Encabezado Fijo.{0,5}", "", texto)

    # 2. LIMPIAR RUIDO DE FÓRMULAS MATEMÁTICAS ROTAS (Capturado de la captura de pantalla)
    # Volar los rombos con signos de pregunta (caracteres no reconocidos)
    texto = texto.replace("", "")
    
    # Eliminar la secuencia 'ssss' o 'sss' aislada que dejó la conversión de símbolos matemáticos
    texto = re.sub(r'\bsss+\b', '', texto)
    
    # Eliminar repeticiones extrañas de letras 'i' solas (ej: iiiii, iiii) generadas por subíndices
    texto = re.sub(r'\b[iI]{3,}\b', '', texto)
    
    # Reemplazar combinaciones rotas obvias por un espacio (ej: "σσKK-SS")
    # Podés agregar acá las combinaciones específicas si querés limpiarlas por completo
    texto = re.sub(r'σσ\w*-\w*', '', texto)

    # 2. LIMPIEZA DE FORMATO (CONSERVANDO ACENTOS Y Ñ)
    # Reemplazamos saltos de línea y múltiples espacios por un solo espacio en blanco
    texto = re.sub(r'\s+', ' ', texto)
    
    # Removemos solo caracteres de control invisibles (saltos de página, campanas, etc.)
    # Esto NO toca letras, acentos ni la Ñ
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)

    return texto.strip()

if __name__ == "__main__":
    ruta_del_script = os.path.dirname(os.path.abspath(__file__))
    carpeta_data = os.path.join(ruta_del_script, "data")
    
    ruta_ingreso = os.path.join(carpeta_data, "corpus_sostenibilidad.json")
    ruta_salida = os.path.join(carpeta_data, "corpus_limpio.json")
    
    # CONFIGURACIÓN DE PÁGINAS A EXCLUIR (Índice)
    # Podés poner los números de página exactos del índice para que el RAG los ignore
    PAGINAS_A_EXCLUIR = [4, 5] 
    
    if not os.path.exists(ruta_ingreso):
        print(f"[ERROR] No se encontró {ruta_ingreso}.")
    else:
        # Forzamos la lectura con encoding UTF-8 para garantizar acentos y Ñ
        with open(ruta_ingreso, "r", encoding="utf-8") as f:
            datos_crudos = json.load(f)
            
        print(f"=== Iniciando Limpieza Avanzada de {len(datos_crudos)} páginas ===")
        datos_limpios = []
        
        for item in datos_crudos:
            num_pag = item["metadata"]["pagina"]
            texto_original = item["contenido"]
            
            # 1. Filtro de Índice por número de página
            if num_pag in PAGINAS_A_EXCLUIR:
                print(f"-> Saltando página {num_pag} por estar en la lista de exclusión (Índice).")
                continue
                
            # 2. Filtro de Índice por palabra clave (por las dudas)
            if "índice" in texto_original.lower() and len(texto_original) < 1500:
                print(f"-> Saltando página {num_pag} porque parece ser un índice de contenidos.")
                continue
            
            # Aplicar la limpieza de encabezados y formato
            texto_procesado = limpiar_texto_avanzado(texto_original)
            
            if len(texto_procesado) > 20:
                item_limpio = {
                    "contenido": texto_procesado,
                    "metadata": item["metadata"]
                }
                datos_limpios.append(item_limpio)
        
        # Guardar el resultado asegurando UTF-8 nativo
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(datos_limpios, f, ensure_ascii=False, indent=4)
            
        print("\n==================================================")
        print("[ÉXITO] ¡Limpieza avanzada completada!")
        print(f"Archivo generado de forma segura: {ruta_salida}")
        print(f"Páginas finales en el corpus: {len(datos_limpios)}")
        print("==================================================")