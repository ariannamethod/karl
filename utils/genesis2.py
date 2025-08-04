import random
import textwrap
from datetime import datetime, timezone
import httpx
import re
from typing import List

from .config import settings  # settings.PPLX_API_KEY должен быть определён
from .message_helper import sentence_end_pattern

# Самая универсальная рабочая модель на сегодня:
PPLX_MODEL = "sonar-pro"
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
TIMEOUT = 25

headers = {
    "Authorization": f"Bearer {settings.PPLX_API_KEY}",
    "Content-Type": "application/json",
}


def _build_prompt(draft: str, user_prompt: str, language: str) -> list:
    system_msg = textwrap.dedent(
        f"""
        You are GENESIS-2, the intuition filter for Indiana‐AM ("Indiana Jones" archetype).
        Return ONE short investigative twist (≤500 tokens) that deepens the current reasoning.
        Do **NOT** repeat the draft; just add an angle, question or hidden variable.
        Reply in {language}.
        """
    ).strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"USER PROMPT >>> {user_prompt}"},
        {"role": "assistant", "content": f"DRAFT >>> {draft}"},
        {"role": "user", "content": "Inject the twist now:"},
    ]


async def _call_sonar(messages: list) -> str:
    payload = {
        "model": PPLX_MODEL,
        "messages": messages,
        "temperature": 0.8,  # регулируй, если нужно разнообразие
        "max_tokens": 500,  # Увеличен лимит токенов с 120 до 500
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(PPLX_API_URL, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except Exception:
            # Дебаг: показать тело ошибки API
            print("[Genesis-2] Sonar HTTP error:", resp.text)
            raise
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()


def split_message(message: str, max_length: int = 4000) -> List[str]:
    """
    Разбивает длинное сообщение на части, сохраняя целостность предложений
    """
    if len(message) <= max_length:
        return [message]
    
    parts = []
    current_text = message
    
    while len(current_text) > 0:
        if len(current_text) <= max_length:
            parts.append(current_text)
            break
        
        # Ищем последний конец предложения или абзаца до max_length
        cut_point = max_length
        last_period = current_text[:cut_point].rfind('. ')
        last_exclamation = current_text[:cut_point].rfind('! ')
        last_question = current_text[:cut_point].rfind('? ')
        last_paragraph = current_text[:cut_point].rfind('\n\n')
        
        end_points = [p for p in [last_period, last_exclamation, last_question, last_paragraph] if p != -1]
        
        if end_points:
            # Берем самую дальнюю точку разрыва
            cut_point = max(end_points) + 2
        else:
            # Если не нашли хорошей точки разрыва, ищем последний пробел
            last_space = current_text[:cut_point].rfind(' ')
            if last_space != -1:
                cut_point = last_space + 1
        
        parts.append(current_text[:cut_point])
        current_text = current_text[cut_point:]
    
    return parts


async def genesis2_sonar_filter(user_prompt: str, draft_reply: str, language: str) -> str:
    # Не всегда срабатывать — для "живости"
    if random.random() < 0.12 or not settings.PPLX_API_KEY:
        return ""
    try:
        messages = _build_prompt(draft_reply, user_prompt, language)
        twist = await _call_sonar(messages)
        
        # Проверка на обрезание сообщения посередине предложения
        if not re.search(sentence_end_pattern, twist):
            twist = re.sub(r'\w+$', '...', twist)
        
        return twist
    except Exception as e:
        print(f"[Genesis-2] Sonar fail {e} @ {datetime.now(timezone.utc).isoformat()}")
        return ""


async def assemble_final_reply(user_prompt: str, indiana_draft: str, language: str) -> str:
    twist = await genesis2_sonar_filter(user_prompt, indiana_draft, language)
    if twist:
        final_reply = f"{indiana_draft}\n\n🜂 Investigative Twist → {twist}"
        
        # Проверяем, нужно ли разделить сообщение
        if len(final_reply) > 4000:
            parts = split_message(final_reply)
            
            # Добавляем индикатор продолжения для всех частей, кроме последней
            for i in range(len(parts) - 1):
                parts[i] = parts[i] + "\n\n[продолжение следует...]"
            
            return parts
        
        return final_reply
    
    return indiana_draft