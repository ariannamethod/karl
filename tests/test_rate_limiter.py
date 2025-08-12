from datetime import datetime

import pytest
from aiogram.types import Chat, Message, User

from utils.rate_limiter import RateLimitMiddleware

@pytest.mark.asyncio
async def test_rate_limiter_blocks_excess_messages():
    middleware = RateLimitMiddleware(limit=2, window=60, delay=0)
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=1, type="private")
    message = Message(message_id=1, date=datetime.now(), chat=chat, from_user=user, text="hello")

    calls = 0

    async def handler(event, data):
        nonlocal calls
        calls += 1

    await middleware(handler, message, {})
    await middleware(handler, message, {})
    await middleware(handler, message, {})

    assert calls == 2
