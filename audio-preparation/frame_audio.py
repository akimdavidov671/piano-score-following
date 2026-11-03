import argparse
import sys
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 8_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes, signed 16-bit PCM
FRAME_LENGTH = 512  # samples, 64 ms
HOP_SIZE = 128  # samples, 16 ms


def load_samples(path: Path) -> np.ndarray:
    """Load validated mono 8-kHz PCM16 WAV into a one-dimensional array."""
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        expected = (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH)
        if wav.getcomptype() != "NONE" or actual != expected:
            raise ValueError(
                f"Expected uncompressed 8000-Hz mono 16-bit PCM, "
                f"got rate={actual[0]}, channels={actual[1]}, "
                f"sample_width={actual[2]}, compression={wav.getcomptype()}"
            )
        num_samples = wav.getnframes()
        pcm_bytes = wav.readframes(num_samples)

    if len(pcm_bytes) != num_samples * SAMPLE_WIDTH:
        raise ValueError(f"WAV audio data is incomplete: {path.name}")

    # WAV PCM16 stores little-endian samples. No scaling or normalization.
    return np.frombuffer(pcm_bytes, dtype="<i2")


def create_frames(samples: np.ndarray):
    """Return frames, start sample indices, and start times (seconds)."""
    if samples.ndim != 1:
        raise ValueError("Expected a one-dimensional sample array")
    if samples.size < FRAME_LENGTH:
        raise ValueError(
            f"Recording has {samples.size} samples: fewer than a "
            f"{FRAME_LENGTH}-sample frame"
        )

    # Each row is a 512-sample VIEW into the original sample array.
    # No sample values are changed. np.savez_compressed writes these rows.
    frames = np.lib.stride_tricks.sliding_window_view(
        samples, window_shape=FRAME_LENGTH
    )[::HOP_SIZE]

    frame_index = np.arange(frames.shape[0], dtype=np.int64)
    start_sample = frame_index * HOP_SIZE
    time_seconds = start_sample / SAMPLE_RATE
    return frames, frame_index, start_sample, time_seconds


def process_one(wav_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Create one NPZ; return (sample_count, frame_count, unused_end_samples)."""
    samples = load_samples(wav_path)
    frames, frame_index, start_sample, time_seconds = create_frames(samples)
    unused = int(samples.size - (int(start_sample[-1]) + FRAME_LENGTH))

    # Write to a temporary file before replacing the destination, so a
    # failed/incomplete write does not leave a misleading output archive.
    temp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temp_path.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                frames=frames,
                frame_index=frame_index,
                start_sample=start_sample,
                time_seconds=time_seconds,
                sample_rate=np.int64(SAMPLE_RATE),
                frame_length=np.int64(FRAME_LENGTH),
                hop_size=np.int64(HOP_SIZE),
                original_samples=np.int64(samples.size),
            )
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return int(samples.size), int(frames.shape[0]), unused


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=project_root / "audio" / "wav",
        help="Directory containing converted .wav files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "audio" / "frames",
        help="Directory to write .npz frame arrays",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: WAV directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: no .wav files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"WAV directory:   {input_dir}")
    print(f"Frame directory: {output_dir}")
    print(f"Found {len(wav_files)} WAV file(s).")
    print(f"Framing: {FRAME_LENGTH} samples, {HOP_SIZE}-sample hop.\n")

    succeeded = failed = 0
    for wav_path in wav_files:
        destination = output_dir / f"{wav_path.stem}.npz"
        try:
            sample_count, n_frames, unused = process_one(wav_path, destination)
            print(
                f"[OK] {wav_path.name} -> {destination.name}: "
                f"{sample_count:,} samples, {n_frames:,} frames, "
                f"{unused} trailing samples unused"
            )
            succeeded += 1
        except (OSError, ValueError, wave.Error) as exc:
            print(f"[ERROR] {wav_path.name}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nSummary: {succeeded} processed, {failed} failed.")
    if len(wav_files) != 50:
        print(f"NOTE: found {len(wav_files)} WAV files; the study uses 50 excerpts.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
