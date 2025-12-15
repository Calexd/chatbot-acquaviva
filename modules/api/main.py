from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import telebot
from . import chat
import os
from modules.telegram_bot import bot_logic # Importamos la lógica del bot

app = FastAPI(title="Acquaviva Chatbot API", description="API RAG Serverless + Telegram Bot")

# --- CONFIGURACIÓN TELEGRAM ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TOKEN)
# Cargamos las reglas de respuesta (handlers)
bot_logic.register_handlers(bot)

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
    # Configurar Webhook automáticamente al iniciar (Opcional pero útil)
    # url_webhook = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/telegram_webhook"
    # bot.remove_webhook()
    # bot.set_webhook(url=url_webhook)

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

# --- NUEVO ENDPOINT PARA TELEGRAM (WEBHOOK) ---
@app.post(f"/telegram_webhook")
async def process_webhook(request: Request):
    """
    Aquí Telegram envía los mensajes nuevos.
    """
    json_str = await request.json()
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)