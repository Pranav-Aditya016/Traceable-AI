import io
import json
import sys

import requests
from PIL import Image, ImageDraw


def build_sample_image_bytes() -> bytes:
    image = Image.new("RGB", (640, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Patient A, age 45, fever and cough for 3 days", fill="black")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_smoke_test(base_url: str, max_events: int = 5) -> int:
    payload = build_sample_image_bytes()

    response = requests.post(
        f"{base_url.rstrip('/')}/api/analyze",
        files={"file": ("sample.png", payload, "image/png")},
        stream=True,
        timeout=600,
    )

    print(f"status={response.status_code}")
    print(f"content_type={response.headers.get('content-type')}")

    if response.status_code != 200:
        print("request failed")
        return 1

    events_seen = 0
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        stage = event.get("stage")
        status = event.get("status")
        label = event.get("label", "")
        result = str(event.get("result", ""))
        print(f"event stage={stage} status={status} label={label} result={result[:120]}")

        events_seen += 1

        if status == "error":
            print("pipeline emitted error")
            response.close()
            return 1

        if stage == 6 and status == "complete":
            print("pipeline complete")
            response.close()
            return 0

        if max_events > 0 and events_seen >= max_events:
            print(f"smoke threshold reached ({max_events} events)")
            response.close()
            return 0

    response.close()
    print("stream ended unexpectedly")
    return 1


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    max_events = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    code = run_smoke_test(url, max_events=max_events)
    sys.exit(code)
