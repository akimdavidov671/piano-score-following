#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np


SAMPLE_RATE = 8_000
FRAME_LENGTH = 512
HOP_SIZE = 128
N_BINS = 256
DEFAULT_WINDOW = "hann_periodic"
DEFAULT_CHUNK_SIZE = 2048


def make_window(kind: str = DEFAULT_WINDOW) -> np.ndarray:
    """Return a float64, 512-sample analysis window.

    'hann_periodic' is the DFT/analysis convention: np.hanning(513)[:-1].
    'hann_symmetric' uses endpoints exactly at zero: np.hanning(512).
    Both are choices to document, not window types specified by Jiang.
    """
    if kind == "hann_periodic":
        return np.hanning(FRAME_LENGTH + 1)[:-1]
    if kind == "hann_symmetric":
        return np.hanning(FRAME_LENGTH)
    raise ValueError(f"Unknown window {kind!r}")


def _validate_2d(array: np.ndarray, n_columns: int, label: str) -> None:
    if array.ndim != 2 or array.shape[1] != n_columns:
        raise ValueError(f"{label} must have shape (n_frames, {n_columns}), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{label} must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinite values")


def apply_window(frames: np.ndarray, *, kind: str = DEFAULT_WINDOW) -> np.ndarray:
    """Weight each frame with a Hann window; return float64 values."""
    frames = np.asarray(frames)
    _validate_2d(frames, FRAME_LENGTH, "frames")
    return frames.astype(np.float64) * make_window(kind)[None, :]


def compute_spectra(windowed_frames: np.ndarray) -> np.ndarray:
    """Return nonnegative magnitudes for DFT frequency bins 0..255.

    rfft supplies bins 0..256 for a 512-point real-input DFT. We omit
    bin 256 (the 4000-Hz Nyquist bin), matching Jiang's 256 components.
    """
    windowed_frames = np.asarray(windowed_frames)
    _validate_2d(windowed_frames, FRAME_LENGTH, "windowed_frames")
    coefficients = np.fft.rfft(windowed_frames, n=FRAME_LENGTH, axis=1)
    return np.abs(coefficients[:, :N_BINS])


def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """L1-normalize each row; keep exactly zero-energy rows at zero.

    A nonzero row has sum 1. The zero-vector convention is intentional:
    it is not a probability distribution and should be handled explicitly
    if the downstream likelihood model encounters silent observations.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    _validate_2d(spectra, N_BINS, "spectra")
    if np.any(spectra < 0):
        raise ValueError("Spectral magnitudes must be nonnegative")
    totals = spectra.sum(axis=1, keepdims=True)
    result = np.zeros_like(spectra, dtype=np.float64)
    np.divide(spectra, totals, out=result, where=(totals > 0))
    return result


def extract_audio_features(
    frames: np.ndarray,
    *,
    window: str = DEFAULT_WINDOW,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Transform frames to normalized 256-bin magnitude spectra.

    Chunks limit the peak temporary FFT memory use for long recordings.
    Intermediate windowed frames and raw spectra are not saved to disk.
    """
    frames = np.asarray(frames)
    _validate_2d(frames, FRAME_LENGTH, "frames")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    make_window(window)  # Validate even for an empty array.

    features = np.empty((frames.shape[0], N_BINS), dtype=np.float64)
    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))
        windowed = apply_window(frames[start:end], kind=window)
        spectra = compute_spectra(windowed)
        features[start:end] = normalize_spectra(spectra)
    return features


def load_frame_archive(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load and verify a frame_audio.py archive and its timing metadata."""
    keys = (
        "frames", "frame_index", "start_sample", "time_seconds",
        "sample_rate", "frame_length", "hop_size", "original_samples",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"Missing keys: {sorted(missing)}")
        data = {key: archive[key] for key in keys}

    frames = data.pop("frames")
    _validate_2d(frames, FRAME_LENGTH, "frames")
    if frames.dtype != np.dtype("int16"):
        raise ValueError(f"Expected 16-bit signed PCM frames, got {frames.dtype}")
    if len(frames) == 0:
        raise ValueError("No audio frames in archive")

    for key, expected in (
        ("sample_rate", SAMPLE_RATE),
        ("frame_length", FRAME_LENGTH),
        ("hop_size", HOP_SIZE),
    ):
        if data[key].ndim != 0 or int(data[key]) != expected:
            raise ValueError(f"Expected {key}={expected}, got {data[key]}")

    n = frames.shape[0]
    indices = np.arange(n, dtype=np.int64)
    start_samples = indices * HOP_SIZE
    if not np.array_equal(data["frame_index"], indices):
        raise ValueError("frame_index is missing values or is out of order")
    if not np.array_equal(data["start_sample"], start_samples):
        raise ValueError("start_sample does not match the 128-sample hop")
    if data["time_seconds"].shape != (n,) or not np.allclose(
        data["time_seconds"], start_samples / SAMPLE_RATE, rtol=0, atol=1e-10
    ):
        raise ValueError("Frame timestamps do not match the start sample indices")

    if data["original_samples"].ndim != 0:
        raise ValueError("original_samples must be a scalar")
    original_samples = int(data["original_samples"])
    if original_samples < FRAME_LENGTH or (
        1 + (original_samples - FRAME_LENGTH) // HOP_SIZE
    ) != n:
        raise ValueError("Frame count does not agree with original_samples")

    return frames, data


def process_one(path: Path, output_path: Path, *, window: str, chunk_size: int) -> tuple[int, int]:
    """Process one archive; return number of frames and exactly silent frames."""
    frames, metadata = load_frame_archive(path)
    features = extract_audio_features(frames, window=window, chunk_size=chunk_size)
    silent_mask = np.all(features == 0.0, axis=1)
    n_silent = int(silent_mask.sum())

    # Verify the essential output invariant before saving.
    non_silent = ~silent_mask
    if not np.all(np.isfinite(features)) or np.any(features < 0):
        raise ValueError("Feature array has invalid magnitude values")
    if non_silent.any() and not np.allclose(
        features[non_silent].sum(axis=1), 1.0, atol=1e-12, rtol=0
    ):
        raise ValueError("One or more non-silent feature vectors do not sum to 1")

    # Atomic replacement, like the other preprocessing scripts.
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                features=features,
                frame_index=metadata["frame_index"],
                start_sample=metadata["start_sample"],
                time_seconds=metadata["time_seconds"],
                sample_rate=metadata["sample_rate"],
                frame_length=metadata["frame_length"],
                hop_size=metadata["hop_size"],
                original_samples=metadata["original_samples"],
                window=np.str_(window),
                fft_method=np.str_("numpy.fft.rfft"),
                fft_size=np.int64(FRAME_LENGTH),
                retained_bin_start=np.int64(0),
                retained_bin_end_exclusive=np.int64(N_BINS),
                normalization=np.str_("L1 magnitude, zero rows stay zero"),
                zero_energy_mask=silent_mask,
                source_frame_file=np.str_(path.name),
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(frames), n_silent


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "audio" / "frames")
    parser.add_argument("--output-dir", type=Path, default=root / "audio" / "features")
    parser.add_argument(
        "--window", choices=("hann_periodic", "hann_symmetric"),
        default=DEFAULT_WINDOW,
        help="Hann convention to use (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Maximum number of frames processed per FFT batch (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be at least 1")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    input_files = sorted(input_dir.glob("*.npz"))
    if not input_files:
        print(f"ERROR: no .npz frame files found in {input_dir}", file=sys.stderr)
        return 1
    if input_dir == output_dir:
        print("ERROR: input and output directories must be different", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Window: {args.window}; FFT: numpy.fft.rfft; bins: 0..255")
    print(f"Found {len(input_files)} frame archive(s).\n")

    successes = failures = silent_total = 0
    for path in input_files:
        target = output_dir / path.name
        try:
            n_frames, n_silent = process_one(
                path, target, window=args.window, chunk_size=args.chunk_size
            )
            print(
                f"[OK] {path.name}: {n_frames:,} frames -> "
                f"{n_frames:,} x {N_BINS} normalized features "
                f"({n_silent} zero-energy frames)"
            )
            successes += 1
            silent_total += n_silent
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            failures += 1
            print(f"[ERROR] {path.name}: {exc}", file=sys.stderr)

    print(f"\nSummary: {successes} processed, {failures} failed; {silent_total} zero-energy frames.")
    if len(input_files) != 50:
        print(f"NOTE: found {len(input_files)} frame files; the study uses 50 excerpts.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
