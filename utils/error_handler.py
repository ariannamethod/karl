import logging
from aiogram.types.error_event import ErrorEvent

logger = logging.getLogger(__name__)

async def error_handler(event: ErrorEvent) -> bool:
    """Handle unexpected errors and notify the user."""
    logger.error("Unhandled error", exc_info=True)
    try:
        if event.update.message:
            await event.update.message.answer(
                "Произошла внутренняя ошибка, мы уже изучаем её"
            )
    except Exception:
        logger.error("Failed to send error notification", exc_info=True)
    return True
