from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


SYSTEM_PROMPT = """Mimari cephe lejant raporu yazan teknik bir asistansın.
Yanıtı Türkçe ver. Sayıları uydurma; sadece verilen JSON verisini kullan.
Belirsiz veya cepheye atanamayan tespitleri ayrıca belirt."""


def load_json(path: Path | None) -> object:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a report with a vLLM OpenAI-compatible server.")
    parser.add_argument("--elements", required=True, type=Path)
    parser.add_argument("--facades", type=Path, default=None)
    parser.add_argument("--base-url", required=True, help="Example: http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = {
        "facades": load_json(args.facades),
        "elements": load_json(args.elements),
    }
    user_prompt = (
        "Aşağıdaki facade segmentation ve element detection çıktılarını kullanarak "
        "kısa, teknik ve denetlenebilir bir mimari cephe lejant raporu hazırla.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        json={
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
