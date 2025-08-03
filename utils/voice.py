from pathlib import Path

from openai import AsyncOpenAI


async def text_to_voice(client: AsyncOpenAI, text: str) -> bytes:
    """Generate speech audio from text using OpenAI TTS."""
    response = await client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="verse",
        input=text,
        response_format="opus",
    )
    return await response.aread()


async def voice_to_text(client: AsyncOpenAI, file_path: Path) -> str:
    """Transcribe speech audio to text using Whisper."""
    with file_path.open("rb") as f:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return response.text
