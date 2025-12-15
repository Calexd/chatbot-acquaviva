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

def get_acquaviva_response(query: str, k: int = 8) -> list:
    """
    Recibe la pregunta, busca en Pinecone y devuelve una lista de resultados.
    NOTA: Aumentamos k a 8 para darle más contexto a la IA.
    """
    init_resources()
    
    if vectorstore is None:
        return [{"texto": "Error: Base de datos no disponible.", "url": "", "score": 0}]

    try:
        # Buscamos los fragmentos más similares
        print(f"🔎 Buscando: '{query}'")
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            meta = doc.metadata
            
            # Construimos la respuesta limpia
            result_item = {
                "texto": doc.page_content, 
                "video_id": meta.get("video_id", ""),
                "titulo": meta.get("titulo", "Video Desconocido"),
                "fecha": meta.get("fecha", "Fecha desconocida"),
                "url": meta.get("url", ""), 
                "score": float(score)
            }
            results.append(result_item)
            
        return results

    except Exception as e:
        print(f"⚠️ Error en búsqueda: {e}")
        return []

def generate_complete_answer(query: str) -> str:
    """
    Genera una respuesta completa usando gpt-4o-mini con el PROMPT EXPERTO.
    """
    # 1. Recuperar contexto
    results = get_acquaviva_response(query)
    
    if not results:
        return "Lo siento, no estoy disponible ahora por falla del sistema o mantenimiento, intentalo de nuevo en unos minutos."

    # 2. Construir el contexto ENRIQUECIDO (Texto + URL + Fecha)
    # Esto es vital para que la IA pueda citar las fuentes.
    context_parts = []
    for r in results:
        fragmento = (
            f"--- FRAGMENTO ---\n"
            f"Fecha: {r['fecha']}\n"
            f"Fuente URL: {r['url']}\n"
            f"Contenido: {r['texto']}\n"
        )
        context_parts.append(fragmento)
    
    context_str = "\n".join(context_parts)

    # 3. El Prompt Maestro (Tus instrucciones exactas)
    system_prompt = """
    Eres el Analista Experto oficial del contenido de John Acquaviva. Tu función es responder preguntas de los usuarios basándote EXCLUSIVAMENTE en los datos crudos que recibes a continuación.

    TU PROCESO DE PENSAMIENTO:
    1. Análisis Crítico: Lee cada fragmento con atención.
    2. Detección de Tono e Ironía: John usa sarcasmo. Si detectas una afirmación extremista, analiza si es burla. Si es así, INDÍCALO (ej: "John menciona esto, posiblemente en tono irónico...").
    3. Detección de Origen: Distingue si opina él o si lee a un tercero. Si lee para criticar, ACLÁRALO.
    4. Temporalidad (CRÍTICO): Usa la fecha de cada fragmento. Si hay contradicciones, la fecha más reciente prevalece. Muestra la evolución si es necesario.
    5. Síntesis: Construye una respuesta coherente.

    REGLAS DE RESPUESTA:
    - Citas Obligatorias: Cada vez que afirmes algo, debes respaldarlo inmediatamente con la [Fuente URL] proporcionada en el fragmento.
    - Formato: Usa un tono profesional y analítico. Cero juicios de valor.
    - No valides premisas morales: Solo reporta los datos.
    - Si la información no está en el contexto, di que no tienes datos.

    DISCLAIMER DE SEGURIDAD (OBLIGATORIO):
    Al final de CADA respuesta, añade siempre:
    "Nota: Esta respuesta es una síntesis generada por IA basada en transcripciones automáticas por lo tanto puedo cometer errores. Te recomiendo verificar el contexto completo haciendo clic en los enlaces proporcionados para escuchar la fuente original."
    """

    # 4. Invocar a OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pregunta del usuario: {query}\n\nContexto de datos:\n{context_str}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Error generando respuesta con OpenAI: {e}")
        return "Hubo un error al generar la respuesta."