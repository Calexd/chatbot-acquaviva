from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import telebot
from . import chat
import os
from modules.telegram_bot import bot_logic

app = FastAPI(title="Acquaviva Chatbot API", description="API RAG Serverless + Telegram Bot")

# --- CONFIGURACIÓN TELEGRAM ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
# Verificación de seguridad para evitar error si no hay token (local)
if TOKEN:
    bot = telebot.TeleBot(TOKEN)
    bot_logic.register_handlers(bot)
else:
    bot = None

# --- CONFIGURACIÓN DE SEGURIDAD (WEBHOOK SECRET) ---
# Aquí definimos el nombre de la "puerta secreta".
# Si no configuras nada en Render, usará "telegram_webhook_secreto_v1" por defecto.
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "telegram_webhook_secreto_v1")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    results: List[Dict[str, Any]]

class BotResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    print("🚀 Arrancando API y Recursos...")
    chat.init_resources()

@app.get("/")
def health_check():
    return {"status": "Online", "service": "Acquaviva Knowledge Base"}

# --- ENDPOINTS EXISTENTES (NO TOCAR) ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
    try:
        raw_results = chat.get_acquaviva_response(request.message)
        return ChatResponse(results=raw_results)
    except Exception as e:
        print(f"Error API: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/bot_response", response_model=BotResponse)
def bot_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
    try:
        answer = chat.generate_complete_answer(request.message)
        return BotResponse(response=answer)
    except Exception as e:
        print(f"Error API /bot_response: {e}")
        return BotResponse(response="Ocurrió un error al procesar tu solicitud.")

# --- NUEVO ENDPOINT SEGURO PARA TELEGRAM ---
# Fíjate que ahora la ruta no es fija, usa la variable WEBHOOK_PATH
@app.post(f"/{WEBHOOK_PATH}")
async def process_webhook(request: Request):
    """
    Aquí Telegram envía los mensajes nuevos.
    """
    if not bot:
        return {"status": "error", "message": "Bot no inicializado"}
        
    json_str = await request.json()
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)