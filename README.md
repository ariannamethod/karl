# Indiana (LIGHTHOUSE)

Indiana-AM is an experimental Telegram assistant inspired by the explorer archetype.
It runs entirely on the OpenAI Assistants API using the `gpt-4o` model ("GPT‑4.1").
All messages are processed through a single assistant instance. Recent chats are
stored in a local SQLite database and conversation logs are appended to
`notes/journal.json`. Any text files placed in `artefacts/` are loaded at startup
and used as additional context.

Random delays are applied before replies (10–40 s in private chats, 2–6 minutes in
groups) and the bot may occasionally send an "afterthought" message an hour or so
later.

## Quick start

```bash
# Clone and run
git clone https://github.com/ariannamethod/Indiana-AM.git
cd Indiana-AM
cp .env.example .env   # add TELEGRAM_BOT_TOKEN, OPENAI_API_KEY and chat IDs
# `.env` will be loaded automatically on startup
# After the first run the assistant ID is stored in `assistants.json`.
# Place reading materials in the `artefacts/` folder.
# Logs are written to `notes/journal.json`.
pip install -r requirements.txt
python main.py
```

## License
MIT — because archaeology of consciousness should stay open.
