import telebot
import os
from dotenv import load_dotenv
from modules.api import chat  # Importamos tu cerebro experto directamente

# Cargar entorno
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TOKEN)

def register_handlers(bot_instance):
    """
    Aquí definimos cómo responde el bot.
    Esta función se llamará desde main.py al iniciar el servidor.
    """
    
    @bot_instance.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        bot_instance.reply_to(message, "¡Hola! Soy la IA experta en contenido de John Acquaviva. Mencióname o responde a mis mensajes para preguntarme algo.")

    # Maneja mensajes de texto (Privados o Menciones en Grupos)
    @bot_instance.message_handler(func=lambda message: True)
    def handle_message(message):
        # Filtro para grupos: ¿Es privado O me están mencionando/respondiendo?
        is_private = message.chat.type == 'private'
        is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot_instance.get_me().id
        is_mention = f"@{bot_instance.get_me().username}" in message.text
        
        # Si es grupo y NO me hablan a mí, ignoro para no ser spam
        if not is_private and not (is_reply_to_bot or is_mention):
            return

        user_id = message.from_user.id
        pregunta = message.text.replace(f"@{bot_instance.get_me().username}", "").strip()
        
        print(f"📩 Mensaje de {user_id} en {message.chat.type}: {pregunta}")

        # Acción "Escribiendo..."
        bot_instance.send_chat_action(message.chat.id, 'typing')

        try:
            # USAMOS LA FUNCIÓN EXPERTA DIRECTAMENTE (Sin requests HTTP extra)
            respuesta = chat.generate_complete_answer(pregunta)
            
            # Telegram soporta Markdown, pero a veces falla con caracteres raros. 
            # Enviamos texto plano o HTML simple si quieres.
            bot_instance.reply_to(message, respuesta)
            
        except Exception as e:
            bot_instance.reply_to(message, "⚠️ Tuve un problema procesando tu pregunta.")
            print(f"❌ Error Bot: {e}")