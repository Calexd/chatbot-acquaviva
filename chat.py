import os
import sys
import pandas as pd
from dotenv import load_dotenv

# LangChain / Pinecone imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "acquaviva-bot"
CSV_PATH = "transcription_final.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"  # Fallback: gpt-3.5-turbo

# Hardcoded Disclaimer
DISCLAIMER = "\n\n> **Nota:** Esta respuesta se basa en transcripciones automáticas. Te recomiendo verificar el contexto escuchando el video original en el enlace proporcionado."

# Global Variables for Caching
df = None
vectorstore = None
llm = None
prompt = None

def init_resources():
    """Initializes global resources if they haven't been loaded yet."""
    global df, vectorstore, llm, prompt
    
    if df is not None:
        return  # Already initialized

    print("Initializing resources...")

    # 1. Load CSV
    try:
        print(f"Loading CSV data from {CSV_PATH} into memory...")
        df = pd.read_csv(CSV_PATH, dtype=str)
        print(f"CSV loaded. Rows: {len(df)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # In a real app, maybe raise or fail, but specifically for now we print and maybe let it fail later or exit if script
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise e

    # 2. Setup Pinecone & LangChain
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY missing.")
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY missing.")
        
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Init VectorStore
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)

    # 3. Define Prompt with Temporal Rules
    template = """You are an expert AI assistant specialized in the content of this YouTuber.
    Mantén la personalidad de 'Experto analista'.
    
    Instrucciones Clave:
    1. Matices: Distingue cuándo habla el autor y cuándo lee a terceros para no atribuirle opiniones falsas.
    2. Contexto Temporal: Presta extrema atención a la fecha del video ([Fecha Video: ...]).
       - Si el youtuber usa palabras como "hoy", "recientemente", "ahora" o "esta semana", DEBES transformarlas a la fecha real del video.
       - Incorrecto: "Dice que hoy hay una crisis..."
       - Correcto: "En el video de abril de 2024, menciona que en ese momento había una crisis..."
    
    Answer the user's question based ONLY on the following context fragments.
    
    The context below contains 'Expanded Context' blocks. The line marked with '>>>' is the direct match.
    
    CRITICAL CITATION RULE:
    If you use information from a context fragment, you MUST cite the source by providing the exact YouTube link provided in the context frame.
    Format the link exactly as: https://youtu.be/VIDEO_ID?t=START_TIME
    Do not make up links. Use the 'Source' field provided.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    print("Resources initialized.")


def get_expanded_context(video_id, start_time, window=3):
    """
    Finds the row in the DataFrame matching video_id and start_time,
    retrieves surrounding rows (±window), and injects the video date.
    """
    global df
    if df is None:
        return ""

    try:
        # Find exact match
        matches = df[(df['video_id'] == video_id) & (df['inicio'] == start_time)]
        
        if matches.empty:
            return f"Context expansion failed: Row not found for {video_id} at {start_time}"
            
        target_idx = matches.index[0]
        
        # Get Date
        video_date = matches.iloc[0]['fecha_publicacion']
        
        # Calculate window bounds
        start_idx = max(0, target_idx - window)
        end_idx = min(len(df), target_idx + window + 1)
        
        # Extract slice
        context_slice = df.iloc[start_idx:end_idx]
        
        # Concatenate text with marker
        expanded_text_lines = []
        for _, row in context_slice.iterrows():
            marker = ">>> " if row.name == target_idx else "    "
            expanded_text_lines.append(f"{marker}{row['texto']}")
            
        # Join lines
        concat_text = "\n".join(expanded_text_lines)
        
        # Inject Date Header as requested
        final_block = f"[Fecha Video: {video_date}] Texto: \"{concat_text}\""
        
        return final_block

    except Exception as e:
        return f"Error expanding context: {e}"

def retrieve_and_format(query, vectorstore_instance, k=5):
    """
    Retrieves documents from Pinecone, expands context using Pandas, and formats for the LLM.
    """
    # 1. Similarity Search
    docs = vectorstore_instance.similarity_search(query, k=k)
    
    formatted_contexts = []
    
    for doc in docs:
        meta = doc.metadata
        vid = meta.get('video_id')
        start = meta.get('inicio')
        
        # 2. Expand Context using Pandas (Includes Date Injection)
        expanded_content = get_expanded_context(vid, start)
        
        # Prepare Citation Link
        try:
            start_seconds = int(float(start))
        except:
            start_seconds = 0
        
        link = f"https://youtu.be/{vid}?t={start_seconds}"
        
        # 3. Build Block
        block = (
            f"--- RESULT ---\n"
            f"Source: {link}\n"
            f"Expanded Context:\n{expanded_content}\n"
        )
        formatted_contexts.append(block)
        
    return "\n\n".join(formatted_contexts)

def get_acquaviva_response(question: str) -> str:
    """
    Main entry point for getting a response from the bot.
    Ensures resources are initialized, then searches and answers.
    """
    init_resources()
    
    try:
        # Custom Retrieval + Generation
        context_str = retrieve_and_format(question, vectorstore, k=5)
        
        # Direct chain invoke manually
        chain_input = {"context": context_str, "question": question}
        messages = prompt.format_messages(**chain_input)
        response = llm.invoke(messages)
        
        # Append Disclaimer
        final_response = response.content + DISCLAIMER
        return final_response
        
    except Exception as e:
        return f"Error processing request: {str(e)}"

def main():
    """Legacy CLI interface"""
    print("Starting CLI mode...")
    init_resources()
    
    print("\nSystem Ready (Pinecone Cloud)! Ask questions. Type 'salir' to exit.\n")
    print("-" * 50)

    while True:
        try:
            query = input("User: ").strip()
            if not query:
                continue
            if query.lower() in ['salir', 'exit', 'quit']:
                print("Goodbye!")
                break
                
            print("\nSearching and Thinking...\n")
            
            response = get_acquaviva_response(query)
            
            print(f"AI: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

