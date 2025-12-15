from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn
from . import chat
import os

app = FastAPI(title="Acquaviva Chatbot API", description="API RAG Serverless")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    print("🚀 Arrancando API...")
    chat.init_resources()

@app.get("/")
def health_check():
    return {"status": "Online", "service": "Acquaviva Knowledge Base"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
    
    try:
        # Llamamos al chat.py que ahora devuelve lista de dicts
        raw_results = chat.get_acquaviva_response(request.message)
        return ChatResponse(results=raw_results)
    except Exception as e:
        print(f"Error API: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

class BotResponse(BaseModel):
    response: str

@app.post("/bot_response", response_model=BotResponse)
def bot_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")

    try:
        # Generamos la respuesta completa server-side
        answer = chat.generate_complete_answer(request.message)
        return BotResponse(response=answer)
    except Exception as e:
        print(f"Error API /bot_response: {e}")
        try:
             #Intento de fallback simple si falla la generacion pero no pinecone, aunque generate_complete_answer ya maneja excepciones
             return BotResponse(response="Ocurrió un error al procesar tu solicitud.")
        except:
             raise HTTPException(status_code=500, detail="Error interno del servidor")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)