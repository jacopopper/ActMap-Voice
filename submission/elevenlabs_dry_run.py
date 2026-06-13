from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib import error, request


API_BASE = "https://api.elevenlabs.io"
DEFAULT_TEXT = "ActMap Voice dry run. The route is verify before speaking."
DEFAULT_OUTPUT_DIR = Path("submission/demo_artifacts")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def http_json(method: str, url: str, *, api_key: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    all_headers = {
        "xi-api-key": api_key,
        "Accept": "application/json",
        **(headers or {}),
    }
    req = request.Request(url, data=body, headers=all_headers, method=method)
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {payload[:1000]}") from exc


def http_bytes(method: str, url: str, *, api_key: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
    all_headers = {
        "xi-api-key": api_key,
        **(headers or {}),
    }
    req = request.Request(url, data=body, headers=all_headers, method=method)
    try:
        with request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {payload[:1000]}") from exc


def choose_voice(api_key: str, explicit_voice_id: str | None) -> tuple[str, str]:
    if explicit_voice_id:
        return explicit_voice_id, "from ELEVENLABS_VOICE_ID"

    response = http_json("GET", f"{API_BASE}/v1/voices", api_key=api_key)
    voices = response.get("voices") if isinstance(response, dict) else None
    if not voices:
        raise RuntimeError("No ElevenLabs voices returned. Set ELEVENLABS_VOICE_ID explicitly.")

    voice = voices[0]
    voice_id = str(voice["voice_id"])
    name = str(voice.get("name", "unnamed voice"))
    return voice_id, f"auto-selected voice {name!r}"


def text_to_speech(api_key: str, voice_id: str, text: str) -> bytes:
    payload: dict[str, Any] = {"text": text}
    model_id = os.environ.get("ELEVENLABS_TTS_MODEL_ID")
    if model_id:
        payload["model_id"] = model_id

    body = json.dumps(payload).encode("utf-8")
    return http_bytes(
        "POST",
        f"{API_BASE}/v1/text-to-speech/{voice_id}",
        api_key=api_key,
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )


def multipart_form(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----actmap-elevenlabs-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {media_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def speech_to_text(api_key: str, audio_path: Path) -> dict[str, Any]:
    body, content_type = multipart_form(
        {
            "model_id": os.environ.get("ELEVENLABS_STT_MODEL_ID", "scribe_v2"),
            "tag_audio_events": "false",
        },
        "file",
        audio_path,
    )
    return http_json(
        "POST",
        f"{API_BASE}/v1/speech-to-text",
        api_key=api_key,
        body=body,
        headers={"Content-Type": content_type},
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def transcript_score(expected: str, observed: str) -> float:
    return SequenceMatcher(None, normalize(expected), normalize(observed)).ratio()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run ElevenLabs TTS -> STT for the ActMap Voice demo.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--min-score", type=float, default=0.72)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY is missing. Add it to .env or export it.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = args.output_dir / "elevenlabs_dry_run.mp3"
    transcript_path = args.output_dir / "elevenlabs_dry_run_stt.json"

    voice_id, voice_source = choose_voice(api_key, os.environ.get("ELEVENLABS_VOICE_ID"))
    print(f"[elevenlabs:voice] voice_id={voice_id} source={voice_source}")

    start = time.perf_counter()
    audio = text_to_speech(api_key, voice_id, args.text)
    tts_ms = (time.perf_counter() - start) * 1000
    audio_path.write_bytes(audio)
    print(f"[elevenlabs:tts] ok bytes={len(audio)} path={audio_path} latency_ms={tts_ms:.0f}")

    start = time.perf_counter()
    transcript = speech_to_text(api_key, audio_path)
    stt_ms = (time.perf_counter() - start) * 1000
    transcript_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

    observed = str(transcript.get("text", "")).strip()
    score = transcript_score(args.text, observed)
    print(f"[elevenlabs:stt] ok path={transcript_path} latency_ms={stt_ms:.0f}")
    print(f"[elevenlabs:stt] transcript={observed!r}")
    print(f"[dry-run] expected={args.text!r}")
    print(f"[dry-run] similarity={score:.3f} threshold={args.min_score:.3f}")

    if score < args.min_score:
        print("[dry-run] failed: transcript similarity below threshold", file=sys.stderr)
        return 1

    print("[dry-run] passed: ElevenLabs TTS and STT are usable for the demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
