import argparse
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 8_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def verify_wav(wav_path: Path, original_pcm: bytes) -> None:
    """Verify the WAV format and that the decoded PCM bytes are unchanged."""
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError(f"Unexpected compression in {wav_path.name}")
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH_BYTES
        ):
            raise ValueError(f"Unexpected WAV format in {wav_path.name}")
        if wav.getnframes() != len(original_pcm) // SAMPLE_WIDTH_BYTES:
            raise ValueError(f"Frame count mismatch in {wav_path.name}")
        if wav.readframes(wav.getnframes()) != original_pcm:
            raise ValueError(f"PCM data mismatch in {wav_path.name}")


def convert_one(raw_path: Path, wav_path: Path, overwrite: bool) -> str:
    pcm = raw_path.read_bytes()
    if not pcm:
        raise ValueError(f"Empty RAW file: {raw_path.name}")
    if len(pcm) % SAMPLE_WIDTH_BYTES:
        raise ValueError(f"Odd byte count for 16-bit PCM: {raw_path.name}")

    if wav_path.exists() and not overwrite:
        verify_wav(wav_path, pcm)
        return "skipped (existing WAV verified)"

    # PCM WAV stores 16-bit integers in little-endian byte order. The source
    # has already been verified to use exactly that representation.
    temp_path = wav_path.with_name(wav_path.name + ".tmp")
    try:
        with wave.open(str(temp_path), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        verify_wav(temp_path, pcm)
        temp_path.replace(wav_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return f"converted ({len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES):.2f} s)"


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=project_root / "audio")
    parser.add_argument("--output-dir", type=Path, default=project_root / "audio" / "wav")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing WAV files")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    raw_files = sorted(input_dir.glob("*.raw"))
    if not raw_files:
        print(f"ERROR: no .raw files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"RAW directory: {input_dir}")
    print(f"WAV directory: {output_dir}")
    print(f"Found {len(raw_files)} RAW file(s).\n")

    converted = skipped = failed = 0
    for raw_path in raw_files:
        wav_path = output_dir / f"{raw_path.stem}.wav"
        try:
            result = convert_one(raw_path, wav_path, args.overwrite)
            print(f"[OK] {raw_path.name} -> {wav_path.name}: {result}")
            if result.startswith("converted"):
                converted += 1
            else:
                skipped += 1
        except (OSError, ValueError, wave.Error) as exc:
            failed += 1
            print(f"[ERROR] {raw_path.name}: {exc}", file=sys.stderr)

    print(f"\nSummary: {converted} converted, {skipped} verified/skipped, {failed} failed.")
    if len(raw_files) != 50:
        print(f"NOTE: found {len(raw_files)} RAW files; the study describes 50 excerpts.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
