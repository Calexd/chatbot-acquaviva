import telebot
import os
import time
from dotenv import load_dotenv
from modules.api import chat

# Cargar entorno
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
if TOKEN:
    bot = telebot.TeleBot(TOKEN)
else:
    bot = None

# --- MEMORIA DE SEGURIDAD (RATE LIMITER) ---
# Estructura: { user_id: [tiempo_ultimo_mensaje, contador_spam] }
user_rate_limit = {}

# CONFIGURACIÓN DE LÍMITES
MAX_MESSAGES_PER_MINUTE = 5  # Máximo 5 preguntas por minuto por persona
BLOCK_TIME = 60              # Segundos de castigo si se excede

def check_spam(user_id):
    """
    Retorna True si el usuario está enviando spam.
    """
    current_time = time.time()
    
    if user_id not in user_rate_limit:
        user_rate_limit[user_id] = [current_time, 1]
        return False
    
    last_time, count = user_rate_limit[user_id]
    
    # Si ha pasado más de 1 minuto, reseteamos su contador
    if current_time - last_time > 60:
        user_rate_limit[user_id] = [current_time, 1]
        return False
    
    # Si está dentro del minuto, aumentamos contador
    user_rate_limit[user_id] = [last_time, count + 1]
    
    # Si se pasó del límite, es SPAM
    if count >= MAX_MESSAGES_PER_MINUTE:
        return True
        
    return False

def register_handlers(bot_instance):
    
    @bot_instance.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        bot_instance.reply_to(message, "¡Hola! Soy la IA de John Acquaviva. Respondo dudas sobre sus videos (Máximo 5 preguntas por minuto para evitar saturación).")

    # --- FILTRO + RATE LIMITER ---
    @bot_instance.message_handler(func=lambda message: True, content_types=['text'])
    def handle_message(message):
        user_id = message.from_user.id
        
        # 1. VERIFICACIÓN ANTI-SPAM
        if check_spam(user_id):
            print(f"🚫 SPAM BLOQUEADO del usuario {user_id}")
            # Opcional: Avisarle (o mejor ignorarlo para no gastar recursos)
            if user_rate_limit[user_id][1] == MAX_MESSAGES_PER_MINUTE + 1:
                bot_instance.reply_to(message, "⚠️ Estás preguntando muy rápido. Espera un minuto.")
            return

        # 2. Filtros de grupo y menciones (Lógica original)
        is_private = message.chat.type == 'private'
        is_reply_to_bot = message.reply_to_message and \
                          message.reply_to_message.from_user.username == bot_instance.get_me().username
        bot_name = bot_instance.get_me().username
        is_mention = f"@{bot_name}" in message.text
        
        if not is_private and not (is_reply_to_bot or is_mention):
            return

        pregunta = message.text.replace(f"@{bot_name}", "").strip()
        
        if not pregunta:
            return

        print(f"📩 [{user_id}] Pregunta: {pregunta}")
        bot_instance.send_chat_action(message.chat.id, 'typing')

        try:
            respuesta = chat.generate_complete_answer(pregunta)
            bot_instance.reply_to(message, respuesta, parse_mode='Markdown')
        except Exception as e:
            bot_instance.reply_to(message, "⚠️ Error temporal del sistema.")
            print(f"❌ Error Bot: {e}")