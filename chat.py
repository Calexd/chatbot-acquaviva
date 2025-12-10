import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Pinecone imports
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "acquaviva-bot"
CSV_PATH = "transcription_final.csv"
EMBEDDING_MODEL = "text-embedding-3-small"

# Global Variables for Caching
df = None
vectorstore = None

def init_resources():
    """Initializes global resources if they haven't been loaded yet."""
    global df, vectorstore
    
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
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise e

    # 2. Setup Pinecone
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
    
    print("Resources initialized (Retrieval Only).")


def get_expanded_text(video_id, start_time, window=3):
    """
    Finds the row in the DataFrame matching video_id and start_time,
    retrieves surrounding rows (±window), and injects the video date.
    Returns the raw combined text.
    """
    global df
    if df is None:
        return ""

    try:
        # Find exact match
        matches = df[(df['video_id'] == video_id) & (df['inicio'] == start_time)]
        
        if matches.empty:
            return "" # Fail silently for raw data
            
        target_idx = matches.index[0]
        
        # Calculate window bounds
        start_idx = max(0, target_idx - window)
        end_idx = min(len(df), target_idx + window + 1)
        
        # Extract slice
        context_slice = df.iloc[start_idx:end_idx]
        
        # Concatenate text
        expanded_text_lines = []
        for _, row in context_slice.iterrows():
            expanded_text_lines.append(row['texto'])
            
        # Join lines
        concat_text = "\n".join(expanded_text_lines)
        
        return concat_text

    except Exception as e:
        print(f"Error expanding context: {e}")
        return ""

def retrieve_raw_chunks(query, vectorstore_instance, k=8):
    """
    Retrieves documents from Pinecone and returns structured data.
    """
    # 1. Similarity Search
    docs = vectorstore_instance.similarity_search(query, k=k)
    
    results = []
    
    for doc in docs:
        meta = doc.metadata
        vid = meta.get('video_id')
        start = meta.get('inicio')
        fecha = meta.get('fecha_publicacion', '')
        
        # 2. Expand Context using Pandas
        expanded_text = get_expanded_text(vid, start)
        
        # Prepare URL
        try:
            start_seconds = int(float(start))
        except:
            start_seconds = 0
        
        link = f"https://youtu.be/{vid}?t={start_seconds}"
        
        # 3. Build Dict
        result_item = {
            "texto": expanded_text if expanded_text else doc.page_content,
            "fecha": fecha,
            "video_id": vid,
            "url": link
        }
        results.append(result_item)
        
    return results

def get_acquaviva_response(question: str) -> list:
    """
    Main entry point.
    Returns a list of dictionaries with raw retrieved data.
    """
    init_resources()
    
    try:
        # Raw Retrieval
        results = retrieve_raw_chunks(question, vectorstore, k=8)
        return results
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return []

def main():
    """Legacy CLI interface"""
    print("Starting CLI mode (Retrieval Only)...")
    init_resources()
    
    print("\nSystem Ready! Ask questions to see retrieved chunks. Type 'salir' to exit.\n")
    print("-" * 50)

    while True:
        try:
            query = input("User: ").strip()
            if not query:
                continue
            if query.lower() in ['salir', 'exit', 'quit']:
                print("Goodbye!")
                break
                
            print("\nSearching...\n")
            
            results = get_acquaviva_response(query)
            
            import json
            print(json.dumps(results, indent=2, ensure_ascii=False))
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
