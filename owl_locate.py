"""
Find an object in an image and draw a box around it.

OWLv2 finds the precise boxes, friday's LLM decides which one you meant:

  1. LLM turns the question ("where is the white mixed foam board") into a
     short search phrase ("white foam board").
  2. OWLv2 searches for that phrase plus a few generic ones and keeps the
     top candidate boxes.
  3. The candidates are drawn on the image with big numbers 1, 2, 3..., and
     each one is also cropped and enlarged onto a numbered sheet.
  4. Both images and the question go to the LLM, which picks a number.
  5. That OWLv2 box is drawn on the output image.

Usage:
    python owl_locate.py <image_path> "where is the white color breadboard"
    python owl_locate.py <image_path>        # prompts for the question

Options:
    --candidates 8    how many OWLv2 boxes the LLM chooses from
    --no-open         don't open the result

Writes <image>_owl.jpg (the answer), plus <image>_candidates.jpg and
<image>_tiles.jpg (the numbered boxes and enlarged crops the LLM saw) next to
the input.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import torch
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from vision_test import FRIDAY_ENV, load_env

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# Generic phrases searched alongside the LLM's phrase, so the object still
# turns up as a candidate when the specific phrase matches it poorly.
GENERIC_PHRASES = ["object", "box", "block", "electronic component"]

# Boxes covering more than this fraction of the image are ignored.
MAX_BOX_AREA = 0.8

# Candidates overlapping an already kept box by more than this IoU are dropped.
NMS_IOU = 0.5

# Candidates scoring below this are not shown to the LLM.
MIN_SCORE = 0.05

# The specific phrase's best boxes are always candidates, however the generic
# phrases score.
SPECIFIC_SLOTS = 3

# Each candidate is also shown to the LLM as an enlarged crop of this size.
TILE_SIZE = 256

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#42d4f4", "#f032e6", "#9a6324", "#808000", "#000075"]

PHRASE_PROMPT = """Turn the user's question into a short noun phrase naming the
object to find in an image, keeping colour and other visual details.
Reply with the phrase only, lowercase, no punctuation.
Examples:
"where is the white color breadboard" -> white breadboard
"can you find my red multimeter on the desk" -> red multimeter"""

PICK_PROMPT = """You get two images.
The first is the full photo with numbered coloured boxes drawn on it. Each
number is written in a filled square at a corner of its box.
The second is a sheet of enlarged crops, one per box, each labelled with the
same number. Use the crops to see small details like colour and markings,
and the full photo to see where things are and how they relate.

The user asks where an object is. Decide which numbered box contains that
object. Pick the box that fits the object best, not a box that only contains part of
it or a much larger area around it.
If none of the boxes contain the object, answer 0."""

PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "box": {"type": "integer"},
    },
    "required": ["reason", "box"],
    "additionalProperties": False,
}


def llm_client() -> tuple[OpenAI, str]:
    env = load_env(FRIDAY_ENV)
    client = OpenAI(
        base_url=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
    )
    return client, env["AZURE_OPENAI_DEPLOYMENT_MAIN"]


def question_to_phrase(client: OpenAI, model: str, question: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": PHRASE_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            },
        ],
        temperature=0,
    )
    return response.output_text.strip().strip('"').lower()


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def pick_box(
    client: OpenAI, model: str, numbered: Image.Image, tiles: Image.Image, question: str
) -> dict:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": PICK_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": to_data_url(numbered), "detail": "high"},
                    {"type": "input_image", "image_url": to_data_url(tiles), "detail": "high"},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "pick_box",
                "schema": PICK_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )
    return json.loads(response.output_text)


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def find_candidates(image: Image.Image, phrases: list[str], limit: int) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    inputs = processor(text=[phrases], images=image, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"OWLv2 inference: {(time.perf_counter() - start) * 1000:.0f} ms on {device}")

    # OWLv2 pads the image to a square (bottom/right); the post-processor
    # accounts for that, and boxes are clipped to the image below.
    width, height = image.size
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=0.0, target_sizes=[(height, width)]
    )[0]

    # Each box gets the score and phrase of its best-matching query.
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()
    boxes = results["boxes"].cpu().clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)

    # OWLv2 often proposes a box covering the whole image. Drop those.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (areas < MAX_BOX_AREA * width * height) & (scores >= MIN_SCORE)
    scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

    # The specific phrase (index 0) gets its best boxes in first; generic
    # phrases score higher in busy scenes and would otherwise fill every slot.
    order = scores.argsort(descending=True).tolist()
    specific = [i for i in order if labels[i] == 0][:SPECIFIC_SLOTS]
    rest = [i for i in order if i not in specific]

    candidates = []
    for i in specific + rest:
        if any(iou(boxes[i], c["tensor"]) > NMS_IOU for c in candidates):
            continue
        candidates.append({
            "tensor": boxes[i],
            "box": tuple(boxes[i].tolist()),
            "score": scores[i].item(),
            "phrase": phrases[labels[i].item()],
        })
        if len(candidates) == limit:
            break

    for c in candidates:
        del c["tensor"]
    return candidates


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arialbd.ttf", max(16, size))
    except OSError:
        return ImageFont.load_default()


def draw_label(draw, xy, text, font, fill):
    x, y = xy
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    draw.rectangle([x, y, x + text_w + 10, y + text_h + 10], fill=fill)
    draw.text((x + 5 - text_box[0], y + 5 - text_box[1]), text, fill="white", font=font)


def draw_numbered(image: Image.Image, candidates: list[dict]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(min(image.size) // 18)
    line_width = max(2, min(image.size) // 200)

    for n, c in enumerate(candidates, start=1):
        color = COLORS[(n - 1) % len(COLORS)]
        draw.rectangle(c["box"], outline=color, width=line_width)

    # Labels go on top of every box so no number gets covered by an outline,
    # and each one moves to another corner if it would hide an earlier label.
    placed = []
    for n, c in enumerate(candidates, start=1):
        color = COLORS[(n - 1) % len(COLORS)]
        text_box = draw.textbbox((0, 0), str(n), font=font)
        w = text_box[2] - text_box[0] + 10
        h = text_box[3] - text_box[1] + 10
        x0, y0, x1, y1 = c["box"]
        spots = [(x0, y0), (x1 - w, y0), (x0, y1 - h), (x1 - w, y1 - h),
                 (x0, y0 - h), (x1 - w, y0 - h), (x0, y1), (x1 - w, y1)]
        spots = [(min(max(0, x), image.width - w), min(max(0, y), image.height - h))
                 for x, y in spots]
        spot = next(
            (s for s in spots if not any(overlaps((*s, s[0] + w, s[1] + h), p) for p in placed)),
            spots[0],
        )
        placed.append((*spot, spot[0] + w, spot[1] + h))
        draw_label(draw, spot, str(n), font, color)

    return image


def overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_tiles(image: Image.Image, candidates: list[dict]) -> Image.Image:
    """A sheet of enlarged crops, one per candidate, numbered like the boxes."""
    columns = min(4, len(candidates))
    rows = (len(candidates) + columns - 1) // columns
    gap = 8
    sheet = Image.new(
        "RGB",
        (columns * (TILE_SIZE + gap) + gap, rows * (TILE_SIZE + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(TILE_SIZE // 7)

    for n, c in enumerate(candidates, start=1):
        x0, y0, x1, y1 = c["box"]
        # A little context around the box helps tell similar objects apart.
        pad_x, pad_y = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        crop = image.crop((
            max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(image.width, x1 + pad_x), min(image.height, y1 + pad_y),
        ))
        scale = TILE_SIZE / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))

        row, col = divmod(n - 1, columns)
        tile_x = gap + col * (TILE_SIZE + gap)
        tile_y = gap + row * (TILE_SIZE + gap)
        sheet.paste(crop, (tile_x + (TILE_SIZE - crop.width) // 2, tile_y + (TILE_SIZE - crop.height) // 2))

        color = COLORS[(n - 1) % len(COLORS)]
        draw.rectangle([tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE], outline=color, width=4)
        draw_label(draw, (tile_x, tile_y), str(n), font, color)

    return sheet


def draw_answer(image: Image.Image, candidate: dict, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(min(image.size) // 30)
    line_width = max(3, min(image.size) // 150)

    x0, y0, x1, y1 = candidate["box"]
    draw.rectangle([x0, y0, x1, y1], outline="red", width=line_width)

    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0] + 10
    text_h = text_box[3] - text_box[1] + 10
    label_x = min(x0, image.width - text_w)
    label_y = y0 - text_h if y0 - text_h > 0 else y0
    draw_label(draw, (label_x, label_y), label, font, "red")

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Find an object with OWLv2 + LLM.")
    parser.add_argument("image")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    question = args.question or input("What should I find? ")
    image = Image.open(image_path).convert("RGB")

    client, model = llm_client()

    phrase = question_to_phrase(client, model, question)
    print(f"Searching for: {phrase!r} + {GENERIC_PHRASES}")

    candidates = find_candidates(image, [phrase] + GENERIC_PHRASES, args.candidates)
    if not candidates:
        print("OWLv2 found nothing.")
        return

    for n, c in enumerate(candidates, start=1):
        box = ", ".join(f"{v:.0f}" for v in c["box"])
        print(f"  [{n}] score {c['score']:.2f} ({c['phrase']})  box ({box})")

    numbered = draw_numbered(image, candidates)
    candidates_path = image_path.with_name(f"{image_path.stem}_candidates.jpg")
    numbered.save(candidates_path, quality=95)

    tiles = draw_tiles(image, candidates)
    tiles.save(image_path.with_name(f"{image_path.stem}_tiles.jpg"), quality=95)

    start = time.perf_counter()
    pick = pick_box(client, model, numbered, tiles, question)
    print(f"LLM pick: {(time.perf_counter() - start) * 1000:.0f} ms")
    print(f"LLM chose box {pick['box']}: {pick['reason']}")

    if not 1 <= pick["box"] <= len(candidates):
        print(f"None of the boxes match. Candidates: {candidates_path}")
        return

    chosen = candidates[pick["box"] - 1]
    out_path = image_path.with_name(f"{image_path.stem}_owl.jpg")
    draw_answer(image, chosen, phrase).save(out_path, quality=95)
    print(f"Saved: {out_path}")
    print(f"Candidates: {candidates_path}")

    if sys.platform == "win32" and not args.no_open:
        os.startfile(out_path)


if __name__ == "__main__":
    main()
