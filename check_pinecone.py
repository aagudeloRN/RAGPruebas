import os
from dotenv import load_dotenv
from pinecone import Pinecone
import json

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "vigilancia-dev-index"

print(f"--- Diagnóstico del Índice de Pinecone: '{INDEX_NAME}' ---")

# Inicializar clientes
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"\n[ERROR] El índice '{INDEX_NAME}' no existe en tu proyecto de Pinecone.")
        print("Por favor, verifica el nombre en tu dashboard de Pinecone y en tu código.")
    else:
        pinecone_index = pc.Index(INDEX_NAME)
        print("\n[OK] Conexión con el índice establecida.")

        # 1. Obtener estadísticas del índice
        print("\n1. Obteniendo estadísticas del índice...")
        stats = pinecone_index.describe_index_stats()
        # FIX: The stats object is not directly JSON serializable.
        # Printing the object directly uses its string representation.
        # The important part is accessing its attributes below.
        print(stats)
        
        # FIX: Access attributes directly on the stats object
        vector_count = 0
        if hasattr(stats, 'total_vector_count'):
            vector_count = stats.total_vector_count
            
        if vector_count > 0:
            print(f"\n[OK] ¡Tu índice contiene {vector_count} vectores!")
        else:
            print("\n[ALERTA] Tu índice parece estar vacío (0 vectores).")
            print("Esto explica por qué las búsquedas no devuelven resultados.")
            print("Verifica que el pipeline de ingesta se haya completado sin errores para los documentos que subiste.")

        # 2. (Opcional) Intentar recuperar un vector específico
        # Cambia el ID al de un documento que sepas que has subido.
        # Por ejemplo, si el último documento tiene ID 5 en PostgreSQL, usa "doc_5_chunk_0".
        if vector_count > 0:
            doc_id_to_check = 4 # <--- CAMBIA ESTE NÚMERO al ID de un documento que exista en tu BD
            vector_id_to_check = f"doc_{doc_id_to_check}_chunk_0"
            
            print(f"\n2. Intentando recuperar el vector de prueba: '{vector_id_to_check}'...")
            try:
                fetch_response = pinecone_index.fetch(ids=[vector_id_to_check], namespace="default")
                # FIX: Access the 'vectors' attribute directly. It's a dictionary.
                if fetch_response.vectors and vector_id_to_check in fetch_response.vectors:
                    print("[OK] Se ha recuperado el vector de prueba exitosamente.")
                    # FIX: The values in the .vectors dictionary are Vector objects, which are not JSON serializable.
                    # We must convert each Vector object to a dictionary using its .to_dict() method.
                    vectors_as_dict = {key: value.to_dict() for key, value in fetch_response.vectors.items()}
                    print(json.dumps(vectors_as_dict, indent=2, ensure_ascii=False))


                else:
                    print(f"[ALERTA] No se pudo encontrar el vector con ID '{vector_id_to_check}'.")
                    print("Esto es normal si no tienes un documento con ese ID. Puedes cambiar el valor de 'doc_id_to_check' en este script para probar con un ID que sí exista en tu base de datos.")
            except Exception as e:
                print(f"[ERROR] Ocurrió un error al intentar recuperar el vector: {e}")

except Exception as e:
    print(f"\n[ERROR] Ocurrió un error general al conectar con Pinecone: {e}")

print("\n--- Fin del Diagnóstico ---")