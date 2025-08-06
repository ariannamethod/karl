"""Async dataset generation client.

This module provides a command line utility for generating text samples by
calling a remote API concurrently. The results are appended to a file in
NDJSON format. The generator is able to resume from where it left off by
counting the number of existing lines in the output file on startup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

import aiohttp


def is_valid_uuid(value: str) -> bool:
    """Return ``True`` if *value* is a valid UUID string.

    This function attempts to construct :class:`uuid.UUID`. Any ``ValueError``
    raised during parsing is interpreted as *value* not being a valid UUID.
    """

    try:
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


async def generate_one(
    index: int,
    session: aiohttp.ClientSession,
    url: str,
    prompt_template: str,
    file_lock: asyncio.Lock,
    outfile: Any,
    semaphore: asyncio.Semaphore,
) -> None:
    """Generate a single sample and append it to *outfile*.

    ``index`` is used to format ``prompt_template``.
    """

    async with semaphore:
        prompt = prompt_template.format(i=index)
        async with session.post(url, json={"prompt": prompt}) as resp:
            resp.raise_for_status()
            data = await resp.json()

    if not is_valid_uuid(data.get("id")):
        raise ValueError("Response ID is not a valid UUID")

    line = json.dumps(data, ensure_ascii=False)
    async with file_lock:
        outfile.write(line + "\n")
        outfile.flush()


async def main() -> None:
    """Entry point for the dataset generation utility."""

    parser = argparse.ArgumentParser(description="Async dataset generator")
    parser.add_argument("--base-url", required=True, help="API base URL")
    parser.add_argument(
        "--num-generations", type=int, required=True, help="Total generations"
    )
    parser.add_argument(
        "--prompt", required=True, help="Prompt template with {i} placeholder"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Concurrency limit"
    )
    parser.add_argument("--output", required=True, help="Output NDJSON path")
    parser.add_argument(
        "--headers",
        type=str,
        default=None,
        help="Optional JSON encoded request headers",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for existing, _ in enumerate(f, start=1):
                pass
    outfile = output_path.open("a", encoding="utf-8")

    if existing >= args.num_generations:
        outfile.close()
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    headers: Optional[Dict[str, str]] = None
    if args.headers:
        headers = json.loads(args.headers)

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            generate_one(
                i,
                session,
                args.base_url,
                args.prompt,
                file_lock,
                outfile,
                semaphore,
            )
            for i in range(existing, args.num_generations)
        ]
        await asyncio.gather(*tasks)

    outfile.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
