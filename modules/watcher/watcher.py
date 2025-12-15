import os
import time
import feedparser
import requests
import yt_dlp  # <--- NUEVA HERRAMIENTA
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Configuración
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "acquaviva-index")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY") 
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID") 

# Canales
CHANNELS = [

    {

        "name": "Canal Principal", 

        "id": "UCuTIiHyNqELu6RNtB4AahAA", 

        "filter_type": "all" 

    },

    {

        "name": "Canal Secundario", 

        "id": "UC9Cuo8h5aYfnpTk6fR624vg", 

        "filter_type": "all"

    },

    {

        "name": "Canal Recortes", 

        "id": "UCrsZ3ySlUSY3gore8tBneRg", 

        "filter_type": "only_long" # <--- ESTO ES LA CLAVE

    }

]

def get_video_duration(video_url):
    """Obtiene la duración del video sin descargarlo."""
    try:
        ydl_opts = {
            'quiet': True, 
            'noplaylist': True,
            'extract_flat': False # Necesitamos info detallada para la duración
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info.get('duration', 0)
    except Exception as e:
        print(f"⚠️ No pude ver la duración: {e}")
        return 0

def get_latest_videos(channel_id):
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    feed = feedparser.parse(rss_url)
    videos = []
    for entry in feed.entries:
        videos.append({
            "id": entry.yt_videoid,
            "title": entry.title,
            "link": entry.link
        })
    return videos

def check_if_exists_in_pinecone(video_id):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(
            vector=[0]*1536,
            top_k=1,
            filter={"video_id": {"$eq": video_id}},
            include_metadata=False
        )
        return len(results['matches']) > 0
    except:
        return False

def trigger_runpod_processing(video_url, video_id):
    print(f"🚀 ENVIANDO A RUNPOD: {video_id}")
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    payload = {"input": {"url": video_url, "video_id": video_id}} # Ya no necesitamos enviar filter_type
    try:
        requests.post(url, json=payload, headers=headers)
    except Exception as e:
        print(f"❌ Error RunPod: {e}")

def main():
    print("👀 Vigilante 2.0 Iniciado...")
    
    for channel in CHANNELS:
        print(f"📡 Revisando: {channel['name']}")
        videos = get_latest_videos(channel['id'])
        
        for video in videos:
            # 1. Chequeo de existencia
            if check_if_exists_in_pinecone(video['id']):
                print(f"   [Existe] {video['title'][:30]}...")
                continue
            
            # 2. FILTRADO INTELIGENTE (Aquí está la magia)
            if channel['filter_type'] == "only_long":
                print(f"   ⏳ Verificando duración de: {video['title'][:20]}...")
                duration = get_video_duration(video['link'])
                
                # REGLA: Menos de 20 mins (1200s) se ignora
                if duration < 1200:
                    print(f"   ❌ [IGNORADO] Es muy corto ({duration}s). Ahorrando dinero.")
                    continue
                else:
                    print(f"   ✅ [ACEPTADO] Es largo ({duration}s).")

            # 3. Si pasó los filtros, despertamos a RunPod
            print(f"🚨 PROCESANDO: {video['title']}")
            trigger_runpod_processing(video['link'], video['id'])
            time.sleep(5)

if __name__ == "__main__":
    main()