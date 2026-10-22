"""
Ask friday's LLM where something is in an image, then draw a box around it.

Usage:
    python locate.py <image_path> "where is the white color breadboard"
    python locate.py <image_path>            # prompts for the question

Writes <image>_boxed.jpg next to the input and opens it.
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from vision_test import FRIDAY_ENV, load_env, to_data_url

SYSTEM_PROMPT = """You locate objects in images.
The user asks where something is. Find it and return its bounding box.
Coordinates are normalized integers from 0 to 1000, where (0, 0) is the
top-left corner of the image and (1000, 1000) is the bottom-right.
Make the box tight around the object.
If the object is not in the image, set found to false and return an empty boxes list.
If there are several matching objects, return one box per object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                },
                "required": ["label", "x_min", "y_min", "x_max", "y_max"],
                "additionalProperties": False,
            },
        },
        "description": {"type": "string"},
    },
    "required": ["found", "boxes", "description"],
    "additionalProperties": False,
}


def locate(image_path: Path, question: str) -> dict:
    env = load_env(FRIDAY_ENV)

    client = OpenAI(
        base_url=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
    )

    response = client.responses.create(
        model=env["AZURE_OPENAI_DEPLOYMENT_MAIN"],
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {
                        "type": "input_image",
                        "image_url": to_data_url(image_path),
                        "detail": "high",
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "locate_result",
                "schema": SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )

    return json.loads(response.output_text)


def draw_boxes(image_path: Path, result: dict) -> Path:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    line_width = max(3, min(width, height) // 150)
    try:
        font = ImageFont.truetype("arial.ttf", max(16, height // 30))
    except OSError:
        font = ImageFont.load_default()

    for box in result["boxes"]:
        x0 = box["x_min"] / 1000 * width
        y0 = box["y_min"] / 1000 * height
        x1 = box["x_max"] / 1000 * width
        y1 = box["y_max"] / 1000 * height

        draw.rectangle([x0, y0, x1, y1], outline="red", width=line_width)

        label = box["label"]
        text_box = draw.textbbox((x0, y0), label, font=font)
        text_h = text_box[3] - text_box[1]
        label_y = y0 - text_h - 8 if y0 - text_h - 8 > 0 else y0
        draw.rectangle(
            [x0, label_y, x0 + (text_box[2] - text_box[0]) + 8, label_y + text_h + 8],
            fill="red",
        )
        draw.text((x0 + 4, label_y + 2), label, fill="white", font=font)

    out_path = image_path.with_name(f"{image_path.stem}_boxed.jpg")
    image.save(out_path, quality=95)
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    question = sys.argv[2] if len(sys.argv) > 2 else input("What should I find? ")

    result = locate(image_path, question)
    print(json.dumps(result, indent=2))

    if not result["found"] or not result["boxes"]:
        print("\nNot found in the image.")
        return

    out_path = draw_boxes(image_path, result)
    print(f"\nSaved: {out_path}")

    if sys.platform == "win32":
        os.startfile(out_path)


if __name__ == "__main__":
    main()
