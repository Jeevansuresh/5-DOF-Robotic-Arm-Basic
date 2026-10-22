"""
Send an image to the same Azure OpenAI deployment that friday uses.

Reads AZURE_OPENAI_* from d:/everything/friday/.env and calls the
Responses API exactly like friday's agents do, plus an input_image part.

Usage: python vision_test.py [image_path] [prompt]
"""

import base64
import mimetypes
import sys
from pathlib import Path

from openai import OpenAI

FRIDAY_ENV = Path(r"d:\everything\friday\.env")


def load_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip().upper()] = value.strip().strip('"').strip("'")
    return env


def to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    image_path = Path(
        sys.argv[1] if len(sys.argv) > 1
        else r"d:\everything\friday\Architechture.jpg"
    )
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."

    env = load_env(FRIDAY_ENV)

    client = OpenAI(
        base_url=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
    )

    response = client.responses.create(
        model=env["AZURE_OPENAI_DEPLOYMENT_MAIN"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": to_data_url(image_path)},
                ],
            },
        ],
    )

    print(f"model: {env['AZURE_OPENAI_DEPLOYMENT_MAIN']}")
    print(f"image: {image_path}\n")
    print(response.output_text)


if __name__ == "__main__":
    main()
