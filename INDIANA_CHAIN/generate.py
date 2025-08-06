"""Generate Chain-of-Thought traces using the Indiana Chain model."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from data_utils import load_dataset
from indiana_core import reason_loop


class SimpleMonitor:
    """Minimal monitor that collects reasoning steps in memory."""

    def __init__(self) -> None:
        self.records: List[Dict[str, str]] = []

    def log(self, prompt: str, output: str) -> None:
        self.records.append({"prompt": prompt, "output": output})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CoT traces with Indiana Chain"
    )
    parser.add_argument("--dataset", required=True, help="Dataset path or HF name")
    parser.add_argument("--output", required=True, help="Where to store JSONL results")
    parser.add_argument("--split", default="train", help="Dataset split when using HF")
    parser.add_argument("--prompt-key", default="question", help="Field containing the prompt")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--max-steps", type=int, default=5, help="Reasoning steps")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    samples = load_dataset(args.dataset, split=args.split)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(samples):
            prompt = item[args.prompt_key]
            mon = SimpleMonitor()
            final = reason_loop(
                prompt,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                monitor=mon,
            )
            json.dump(
                {"prompt": prompt, "cot": mon.records, "final": final},
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            if args.num_samples is not None and i + 1 >= args.num_samples:
                break


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
