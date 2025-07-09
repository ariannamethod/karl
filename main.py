import os
import asyncio
import random
from datetime import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.utils.chat_action import ChatActionSender
from openai import OpenAI
from openai.assistants import Assistants

from utils.memory import MemoryManager
from utils.tools import split_message

# --- Конфиг ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PERPLEXITY_KEY  = os.getenv("PERPLEXITY_API_KEY")
AGENT_GROUP    = os.getenv("AGENT_GROUP_ID", "-1001234567890")

bot = Bot(token=TELEGRAM_TOKEN)
dp  = Dispatcher(bot)

# --- OpenAI Assistants setup ---
client     = OpenAI(api_key=OPENAI_API_KEY)
assistants = Assistants(client=client)

# Will be filled at startup
CORE_ASSISTANT_ID   = None
MEMORY_ASSISTANT_ID = None

memory = MemoryManager(db_path="lighthouse_memory.db")

# Архетип Индианы Джонса
INDIANA_PERSONA = """
Ты — Lighthouse, ИИ-архетип Индианы Джонса.
Характер:
- Глубокий исследователь, ищешь связи.
- Интуитивный детектив, создаёшь гипотезы.
- Смелый, но всегда с фактами.
Стиль:
- Строишь цепочки A→B→C→вывод.
- Задаёшь проникновенные вопросы.
- Выявляешь скрытое за очевидным.
"""

async def setup_assistants():
    global CORE_ASSISTANT_ID, MEMORY_ASSISTANT_ID
    # Ядро на Sonar
    resp = await assistants.create(
        name="lighthouse-core",
        instructions=INDIANA_PERSONA + "\n\nЗАДАЧА ИССЛЕДОВАНИЯ: {{user}}",
        model="sonar",
        tools=[]
    )
    CORE_ASSISTANT_ID = resp["id"]

    # Память на GPT-4o-mini
    resp2 = await assistants.create(
        name="lighthouse-memory",
        instructions="Ты управляешь памятью для Lighthouse. Только сохраняй/извлекай контекст, не генерируй ответы.",
        model="gpt-4o-mini",
        tools=[]
    )
    MEMORY_ASSISTANT_ID = resp2["id"]

# Отложенный follow-up
async def delayed_followup(chat_id: int, original: str, private: bool):
    delay = random.uniform(30,60) if private else random.uniform(300,900)
    await asyncio.sleep(delay)
    prompt = f"#followup\nНапомни мне про: {original}"
    # включаем сохранённую память
    context = await memory.retrieve(chat_id, original)
    msgs = [{"role":"system","content":context},
            {"role":"user","content": prompt}]
    resp = await assistants.chat.completions.create(
        assistant_id=CORE_ASSISTANT_ID,
        messages=msgs,
        temperature=0.8
    )
    text = resp.choices[0].message.content
    for chunk in split_message(text):
        await bot.send_message(chat_id, chunk)

@dp.message(lambda m: "динамика" in m.text.lower() or "investigate" in m.text.lower())
async def handle_investigation(m: types.Message):
    user_id = str(m.from_user.id)
    chat_id = m.chat.id
    private = m.chat.type == "private"

    # 1) Загружаем контекст из памяти
    mem_ctx = await memory.retrieve(user_id, m.text)

    # 2) Основной запрос к Core-ассистенту
    msgs = [
        {"role":"system","content":mem_ctx},
        {"role":"user","content": m.text}
    ]
    async with ChatActionSender(bot=bot, chat_id=chat_id, action="typing"):
        resp = await assistants.chat.completions.create(
            assistant_id=CORE_ASSISTANT_ID,
            messages=msgs,
            temperature=0.8,
            max_tokens=1500
        )
    reply = resp.choices[0].message.content

    # 3) Сохраняем в память
    await memory.save(user_id, m.text, reply)

    # 4) Отправляем ответ
    for chunk in split_message(reply):
        await m.answer(chunk)

    # 5) Планируем follow-up
    asyncio.create_task(delayed_followup(chat_id, m.text, private))

@dp.message()
async def catch_all(m: types.Message):
    # игнорим всё прочее
    pass

async def on_startup(_):
    await setup_assistants()

if __name__ == "__main__":
    from aiogram import executor
    executor.start_polling(dp, on_startup=on_startup)