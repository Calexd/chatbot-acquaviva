import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "acquaviva-index") 
EMBEDDING_MODEL = "text-embedding-3-small"

vectorstore = None

def init_resources():
    global vectorstore
    if vectorstore is not None:
        return

    print("🔄 Conectando a Pinecone...")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key: return

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=api_key)
        print("✅ Conexión exitosa.")
    except Exception as e:
        print(f"❌ Error: {e}")

def get_acquaviva_response(query: str, k: int = 8) -> list:
    init_resources()
    if vectorstore is None: return []

    try:
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_and_scores:
            meta = doc.metadata
            # AQUÍ ESTÁ LA CLAVE: Recuperamos el 'orador' de los metadatos
            # Si no existe etiqueta, asumimos 'Video' para no culpar a John falsamente.
            orador = meta.get("orador", "Desconocido") 
            
            results.append({
                "texto": doc.page_content, 
                "titulo": meta.get("titulo", "Video"),
                "fecha": meta.get("fecha", "?"),
                "url": meta.get("url", "#"),
                "orador": orador, # <--- Nuevo campo vital
                "score": float(score)
            })
        return results
    except Exception as e:
        print(f"⚠️ Error búsqueda: {e}")
        return []

def generate_complete_answer(query: str) -> str:
    # 1. Búsqueda Server-side (k=40)
    results = get_acquaviva_response(query, k=40)
    
    if not results:
        return "Lo siento, no tengo información sobre eso en la base de datos."

    # 2. Construir contexto CON NOMBRE DEL ORADOR
    context_parts = []
    for r in results:
        # Le decimos a la IA explícitamente quién habla en cada frase
        context_parts.append(
            f"--- FRAGMENTO ---\n"
            f"Orador: {r['orador']}\n"
            f"Fecha: {r['fecha']} | URL: {r['url']}\n"
            f"Contenido: {r['texto']}\n"
        )
    
    context_str = "\n".join(context_parts)

    # 3. PROMPT ANTICONFUSIÓN
    system_prompt = """
    Eres el Asistente IA de John Acquaviva.
    
    IMPORTANTE SOBRE LOS ORADORES:
    Recibirás transcripciones donde se indica el 'Orador'.
    - Si el Orador es 'John Acquaviva' (o similar), es la opinión directa de John.
    - Si el Orador es OTRO (ej: 'Invitado', 'Video Reacción', 'Entrevistado'), NO atribuyas esa opinión a John.
    - Debes aclarar: "Un invitado mencionó..." o "En un video que John analizaba, se dijo...".
    
    ESTILO DE RESPUESTA:
    - Periodístico, directo y estructurado.
    - Usa **Negritas** para resaltar ideas.
    - Usa Emojis (📌, 🗣️, ⚠️) con moderación.
    
    REGLAS DE LINKS:
    - Cita obligatoria en formato Markdown: `[Ver video 🎥](URL)`.
    
    CONTENIDO:
    - Si John critica algo pero un invitado lo apoya, haz la distinción clara.
    - Prioriza la fecha más reciente de John.

    DISCLAIMER:
    "_Respuesta generada por IA. Verifica el contexto en los links._"
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pregunta: {query}\n\nContexto Clasificado:\n{context_str}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "Hubo un error generando la respuesta."