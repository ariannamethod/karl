from dataclasses import dataclass
import os

@dataclass
class Settings:
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "indiana")
    PINECONE_ENV: str = os.getenv("PINECONE_ENV", "")
    BASE_WEBHOOK_URL: str = os.getenv("BASE_WEBHOOK_URL", "")
    PORT: int = int(os.getenv("PORT", 8080))
    AGENT_GROUP: str = os.getenv("AGENT_GROUP_ID", "-1001234567890")
    GROUP_CHAT: str = os.getenv("GROUP_CHAT", "")
    CREATOR_CHAT: str = os.getenv("CREATOR_CHAT", "")
    PPLX_API_KEY: str = os.getenv("PPLX_API_KEY", os.getenv("PERPLEXITY_API_KEY", ""))
    LOG_PRIVATE_CHATS: bool = os.getenv("LOG_PRIVATE_CHATS", "true").lower() != "false"
    ANONYMIZE_PRIVATE_DATA: bool = os.getenv("ANONYMIZE_PRIVATE_DATA", "false").lower() == "true"
    MAX_MEMORY_RECORDS: int = int(os.getenv("MAX_MEMORY_RECORDS", "1000"))

settings = Settings()
