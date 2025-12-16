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
            results.append({
                "texto": doc.page_content, 
                "titulo": meta.get("titulo", "Video"),
                "fecha": meta.get("fecha", "?"),
                "url": meta.get("url", "#"), 
                "score": float(score)
            })
        return results
    except Exception as e:
        print(f"⚠️ Error búsqueda: {e}")
        return []

def generate_complete_answer(query: str) -> str:
    # 1. Búsqueda con K ALTO para el Bot (Server-side)
    results = get_acquaviva_response(query, k=40)
    
    if not results:
        return "Lo siento, no tengo información sobre eso en la base de datos."

    # 2. Construir contexto
    context_parts = []
    for r in results:
        context_parts.append(f"- Fecha: {r['fecha']} | URL: {r['url']}\n  Texto: {r['texto']}")
    
    context_str = "\n".join(context_parts)

    # 3. PROMPT "PERIODISTA DIGITAL" (Estructura Visual y Markdown)
    system_prompt = """
    Eres el Asistente IA de John Acquaviva. Tu objetivo es responder dudas usando SU contenido, pero no dar juicios de valor, 
    ni tienes la última palabra ya que la información puede contener errores asi que invita siempre a ver la fuente original.
    
    DATOS: Recibes transcripciones con Fecha y URL.
    
    ESTILO DE RESPUESTA (IMPORTANTE):
    - NO uses bloques de texto gigantes. Nadie lee eso.
    - Usa **Negritas** para resaltar las ideas clave o frases contundentes de John.
    - Usa Emojis de forma moderada para listar puntos (ej: 📌, 🗣️, 📅, ⚠️).
    - Estructura tu respuesta en párrafos cortos o listas (bullet points).
    
    REGLAS DE LINKS:
    - Es OBLIGATORIO citar las fuentes.
    - Usa formato Markdown estricto para los links.
    - Formato correcto: `[Ver video 🎥](URL)` o `[Fuente 🔗](URL)`.
    - Nunca pongas la URL suelta. Intégrala en el texto o al final de la frase.

    REGLAS DE CONTENIDO:
    - Detecta ironía y sarcasmo. Si John se burla, dilo.
    - Prioriza la opinión más reciente (mira las fechas).
    - Si critica y luego apoya, explica ese cambio cronológicamente.

    EJEMPLO DE FORMATO DESEADO:
    "Según los videos más recientes, la postura de John es clara:
    
    📌 **Sobre el tema X:** Opina que es un error estratégico.
    🗣️ Mencionó que 'es una locura' confiar en ellos `[Ver video 🎥](URL)`.
    
    Sin embargo, en 2023 pensaba diferente, lo que muestra una evolución..."

    DISCLAIMER FINAL:
    Al final, añade en cursiva:
    "_Respuesta generada por IA, puedo cometer errores. Verifica el contexto en los links._"
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{context_str}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "Hubo un error generando la respuesta."