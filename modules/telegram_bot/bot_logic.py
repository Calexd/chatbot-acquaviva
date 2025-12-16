import telebot
import os
from dotenv import load_dotenv
from modules.api import chat

# Cargar entorno
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
# Verificación de seguridad
if TOKEN:
    bot = telebot.TeleBot(TOKEN)
else:
    bot = None

def register_handlers(bot_instance):
    """
    Definimos cómo responde el bot.
    """
    
    @bot_instance.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        bot_instance.reply_to(message, "¡Hola! Soy la IA de John Acquaviva. Mencióname o responde a mis mensajes para preguntarme algo.")

    # --- FILTRO DE SEGURIDAD ---
    # content_types=['text'] -> Ignora fotos, audios, stickers, videos.
    @bot_instance.message_handler(func=lambda message: True, content_types=['text'])
    def handle_message(message):
        # 1. Filtro para grupos
        is_private = message.chat.type == 'private'
        
        is_reply_to_bot = message.reply_to_message and \
                          message.reply_to_message.from_user.username == bot_instance.get_me().username
        
        bot_name = bot_instance.get_me().username
        is_mention = f"@{bot_name}" in message.text
        
        if not is_private and not (is_reply_to_bot or is_mention):
            return

        user_id = message.from_user.id
        # Limpieza del mensaje
        pregunta = message.text.replace(f"@{bot_name}", "").strip()
        
        if not pregunta:
            return

        print(f"📩 Texto recibido de {user_id}: {pregunta}")

        bot_instance.send_chat_action(message.chat.id, 'typing')

        try:
            # Llamamos a la IA con el nuevo Prompt Híbrido
            respuesta = chat.generate_complete_answer(pregunta)
            
            # Respondemos con Markdown
            bot_instance.reply_to(message, respuesta, parse_mode='Markdown')
            
        except Exception as e:
            bot_instance.reply_to(message, "⚠️ Tuve un problema procesando tu pregunta.")
            print(f"❌ Error Bot: {e}")