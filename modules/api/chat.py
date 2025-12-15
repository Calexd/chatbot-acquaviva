import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "acquaviva-index") # Asegúrate que coincida con tu .env
EMBEDDING_MODEL = "text-embedding-3-small"

vectorstore = None

def init_resources():
    """Inicializa la conexión a Pinecone una sola vez."""
    global vectorstore
    
    if vectorstore is not None:
        return # Ya estaba listo

    print("🔄 Conectando a Pinecone (Modo Serverless)...")
    
    api_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not openai_key:
        print("❌ Error: Faltan las API Keys.")
        return

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=api_key
        )
        print("✅ Conexión a Pinecone exitosa.")
    except Exception as e:
        print(f"❌ Error conectando a Pinecone: {e}")

def get_acquaviva_response(query: str, k: int = 6) -> list:
    """
    Recibe la pregunta, busca en Pinecone y devuelve una lista de resultados.
    """
    init_resources()
    
    if vectorstore is None:
        return [{"texto": "Error: Base de datos no disponible.", "url": "", "score": 0}]

    try:
        # Buscamos los k fragmentos más similares
        print(f"🔎 Buscando: '{query}'")
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            meta = doc.metadata
            
            # Construimos la respuesta limpia
            result_item = {
                "texto": doc.page_content, # El chunk completo (ya viene con contexto)
                "video_id": meta.get("video_id", ""),
                "titulo": meta.get("titulo", "Video Desconocido"),
                "fecha": meta.get("fecha", ""),
                "url": meta.get("url", ""), # El link al segundo exacto
                "score": float(score)
            }
            results.append(result_item)
            
        return results

    except Exception as e:
        print(f"⚠️ Error en búsqueda: {e}")
        return []

if __name__ == "__main__":
    # Prueba rápida local
    print("Probando chat.py...")
    res = get_acquaviva_response("¿Qué opina John del socialismo?")
    for r in res:
        print(f"- {r['titulo']}: {r['texto'][:100]}...")