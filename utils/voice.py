from openai import AsyncOpenAI

async def text_to_voice(client: AsyncOpenAI, text: str) -> bytes:
    """Generate speech audio from text using OpenAI TTS."""
    response = await client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="ogg",
    )
    return await response.aread()
