from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elevenlabs_dry_run import choose_voice, load_dotenv, text_to_speech


DEFAULT_NARRATION = (
    "ElevenLabs transcribes the customer: there is an unauthorized upgrade charge "
    "on the April bill. The transcript-only language model sees billing and routes "
    "to verify, but that misses the operational risk. ActMap renders the local "
    "Qwen activation trajectory as a twelve by thirty two by one twenty eight map. "
    "The vision router locks on escalate with ninety nine point one six percent "
    "confidence, and hands the call to a human before any speech is generated, "
    "while the screen shows the handoff."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the 30-second ElevenLabs voiceover.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--text", default=DEFAULT_NARRATION)
    parser.add_argument("--output", type=Path, default=Path("submission/demo_artifacts/actmap_voiceover.mp3"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("submission/demo_artifacts/actmap_voiceover.json")
    )
    return parser.parse_args()


def mp3_duration_seconds(path: Path) -> float:
    data = path.read_bytes()
    index = 0
    if data.startswith(b"ID3") and len(data) >= 10:
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        index = 10 + size

    bitrates = {
        ("1", 1): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
        ("1", 2): [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
        ("1", 3): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
        ("2", 1): [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
        ("2", 2): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        ("2", 3): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    }
    sample_rates = {
        3: [44100, 48000, 32000],
        2: [22050, 24000, 16000],
        0: [11025, 12000, 8000],
    }
    layer_by_bits = {3: 1, 2: 2, 1: 3}
    version_name = {3: "1", 2: "2", 0: "2"}

    duration = 0.0
    frames = 0
    while index + 4 <= len(data):
        header = int.from_bytes(data[index : index + 4], "big")
        if (header & 0xFFE00000) != 0xFFE00000:
            index += 1
            continue

        version_bits = (header >> 19) & 0b11
        layer_bits = (header >> 17) & 0b11
        bitrate_index = (header >> 12) & 0b1111
        sample_index = (header >> 10) & 0b11
        padding = (header >> 9) & 0b1

        if version_bits == 1 or layer_bits == 0 or bitrate_index in {0, 15} or sample_index == 3:
            index += 1
            continue

        layer = layer_by_bits[layer_bits]
        version = version_name[version_bits]
        bitrate = bitrates[(version, layer)][bitrate_index] * 1000
        sample_rate = sample_rates[version_bits][sample_index]

        if layer == 1:
            frame_size = int((12 * bitrate / sample_rate + padding) * 4)
            samples = 384
        elif layer == 3 and version_bits != 3:
            frame_size = int(72 * bitrate / sample_rate + padding)
            samples = 576
        else:
            frame_size = int(144 * bitrate / sample_rate + padding)
            samples = 1152

        if frame_size <= 0:
            index += 1
            continue

        duration += samples / sample_rate
        frames += 1
        index += frame_size

    if frames == 0:
        raise RuntimeError(f"Could not parse MP3 frames in {path}")
    return duration


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    import os

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY is missing. Add it to .env or export it.", file=sys.stderr)
        return 2

    voice_id, voice_source = choose_voice(api_key, os.environ.get("ELEVENLABS_VOICE_ID"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    audio = text_to_speech(api_key, voice_id, args.text)
    args.output.write_bytes(audio)
    duration = mp3_duration_seconds(args.output)

    manifest = {
        "audio": str(args.output),
        "duration_seconds": duration,
        "target_seconds": 30,
        "text": args.text,
        "voice_id": voice_id,
        "voice_source": voice_source,
        "bytes": len(audio),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[elevenlabs:tts] path={args.output} bytes={len(audio)} "
        f"duration_seconds={duration:.1f} voice={voice_source}"
    )
    if abs(duration - 30.0) > 2.5:
        print(
            f"[voiceover] warning: duration is {duration:.1f}s, outside the 30s target window",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
