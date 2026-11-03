#!/usr/bin/env python3

import argparse
import array
import json
import math
import shutil
import statistics
import subprocess
import sys
import wave
from pathlib import Path


CANDIDATES = [
    ("8-bit unsigned PCM", 1, 1),
    ("8-bit signed PCM", 1, 1),

    ("16-bit signed PCM, little-endian", 2, 1),
    ("16-bit signed PCM, big-endian", 2, 1),

    ("24-bit signed PCM, little-endian", 3, 1),
    ("24-bit signed PCM, big-endian", 3, 1),

    ("32-bit signed PCM, little-endian", 4, 1),
    ("32-bit signed PCM, big-endian", 4, 1),

    ("32-bit float PCM, little-endian", 4, 1),
    ("32-bit float PCM, big-endian", 4, 1),

    ("8-bit unsigned PCM stereo", 1, 2),
    ("16-bit signed PCM stereo, little-endian", 2, 2),
    ("16-bit signed PCM stereo, big-endian", 2, 2),
    ("24-bit signed PCM stereo, little-endian", 3, 2),
    ("24-bit signed PCM stereo, big-endian", 3, 2),
    ("32-bit signed PCM stereo, little-endian", 4, 2),
    ("32-bit signed PCM stereo, big-endian", 4, 2),
]


def run_command(command):
    """Run an external command if available."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def inspect_with_file(path):
    if shutil.which("file") is None:
        return "Not available ('file' command not installed)"

    code, out, err = run_command(["file", "-b", str(path)])

    if code == 0:
        return out

    return f"Failed: {err}"


def inspect_with_ffprobe(path):
    if shutil.which("ffprobe") is None:
        return None, "ffprobe not installed"

    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]

    code, out, err = run_command(command)

    if code != 0 or not out:
        return None, "ffprobe could not identify the file"

    try:
        return json.loads(out), None
    except json.JSONDecodeError:
        return None, "ffprobe returned invalid JSON"


def human_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def analyze_candidates(size_bytes, sample_rate, known_duration=None):
    results = []

    for name, bytes_per_sample, channels in CANDIDATES:
        bytes_per_time_sample = bytes_per_sample * channels

        implied_duration = (
            size_bytes /
            (sample_rate * bytes_per_time_sample)
        )

        result = {
            "name": name,
            "bytes_per_sample": bytes_per_sample,
            "channels": channels,
            "implied_duration": implied_duration
        }

        if known_duration is not None:
            difference = abs(implied_duration - known_duration)
            relative_error = difference / known_duration

            result["difference"] = difference
            result["relative_error"] = relative_error

        results.append(result)

    if known_duration is not None:
        results.sort(key=lambda x: x["relative_error"])

    return results


def print_ffprobe(data):
    print("\n=== FFPROBE ===")

    if not data:
        return

    streams = data.get("streams", [])

    if not streams:
        print("No recognizable streams.")
        return

    for i, stream in enumerate(streams):
        print(f"Stream {i}:")
        print(f"  codec_type:   {stream.get('codec_type')}")
        print(f"  codec_name:   {stream.get('codec_name')}")
        print(f"  sample_rate:  {stream.get('sample_rate')}")
        print(f"  channels:     {stream.get('channels')}")
        print(f"  sample_fmt:   {stream.get('sample_fmt')}")
        print(f"  duration:     {stream.get('duration')}")


def decode_s16(raw_bytes, source_endian):
    """
    Decode raw bytes as signed 16-bit mono PCM.

    Returns an array('h') whose integer values are correct on the current machine.
    """
    usable_len = len(raw_bytes) - (len(raw_bytes) % 2)

    if usable_len != len(raw_bytes):
        print(
            f"WARNING: raw file contains {len(raw_bytes)} bytes, which is odd. "
            "Ignoring the final byte for 16-bit decoding."
        )

    samples = array.array("h")
    samples.frombytes(raw_bytes[:usable_len])

    # array('h') interprets bytes using the machine's native byte order.
    if sys.byteorder != source_endian:
        samples.byteswap()

    return samples


def wav_bytes_from_samples(samples):
    """
    Return little-endian signed 16-bit bytes suitable for a PCM WAV file.
    """
    out = array.array("h", samples)

    if sys.byteorder != "little":
        out.byteswap()

    return out.tobytes()


def write_pcm16_wav(path, samples, sample_rate):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(wav_bytes_from_samples(samples))


def sample_statistics(samples, sample_rate):
    n = len(samples)

    if n == 0:
        return None

    total = 0.0
    total_sq = 0.0
    min_value = 32767
    max_value = -32768
    clipped = 0
    zero_crossings = 0

    delta_abs_sum = 0.0
    delta_sq_sum = 0.0

    previous = None

    for value in samples:
        v = int(value)

        total += v
        total_sq += v * v

        if v < min_value:
            min_value = v
        if v > max_value:
            max_value = v

        if v in (-32768, 32767):
            clipped += 1

        if previous is not None:
            if (previous < 0 <= v) or (previous >= 0 > v):
                zero_crossings += 1

            delta = v - previous
            delta_abs_sum += abs(delta)
            delta_sq_sum += delta * delta

        previous = v

    mean = total / n
    rms = math.sqrt(total_sq / n)

    variance = max(0.0, total_sq / n - mean * mean)
    std = math.sqrt(variance)

    if n > 1:
        mean_abs_delta = delta_abs_sum / (n - 1)
        delta_rms = math.sqrt(delta_sq_sum / (n - 1))
        zcr = zero_crossings / (n - 1)
    else:
        mean_abs_delta = 0.0
        delta_rms = 0.0
        zcr = 0.0

    peak = max(abs(min_value), abs(max_value))

    # Lower values normally indicate a smoother waveform.
    # This is only a heuristic and should not be treated as proof.
    smoothness_ratio = delta_rms / rms if rms > 0 else float("inf")

    return {
        "samples": n,
        "duration": n / sample_rate,
        "min": min_value,
        "max": max_value,
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "clipped": clipped,
        "clipped_pct": 100.0 * clipped / n,
        "zero_crossing_rate": zcr,
        "mean_abs_delta": mean_abs_delta,
        "delta_rms": delta_rms,
        "smoothness_ratio": smoothness_ratio,
    }


def print_statistics(label, stats):
    print(f"\n--- {label} ---")
    print(f"Samples:               {stats['samples']:,}")
    print(f"Duration:              {stats['duration']:.6f} s")
    print(f"Minimum sample:        {stats['min']}")
    print(f"Maximum sample:        {stats['max']}")
    print(f"Mean / DC offset:      {stats['mean']:.3f}")
    print(f"Standard deviation:    {stats['std']:.3f}")
    print(f"RMS amplitude:         {stats['rms']:.3f}")
    print(f"Peak amplitude:        {stats['peak']}")
    print(
        f"Clipped samples:       {stats['clipped']:,} "
        f"({stats['clipped_pct']:.5f}%)"
    )
    print(f"Zero-crossing rate:    {stats['zero_crossing_rate']:.5f}")
    print(f"Mean |sample delta|:   {stats['mean_abs_delta']:.3f}")
    print(f"Delta RMS:             {stats['delta_rms']:.3f}")
    print(f"Delta RMS / RMS:       {stats['smoothness_ratio']:.5f}")


def create_16bit_candidates(path, sample_rate, output_dir):
    raw_bytes = path.read_bytes()

    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = {}

    for endian, suffix in [
        ("little", "s16le"),
        ("big", "s16be"),
    ]:
        samples = decode_s16(raw_bytes, endian)
        stats = sample_statistics(samples, sample_rate)

        wav_path = output_dir / f"{path.stem}_{suffix}.wav"
        write_pcm16_wav(wav_path, samples, sample_rate)

        candidates[suffix] = {
            "endian": endian,
            "samples": samples,
            "stats": stats,
            "wav_path": wav_path,
        }

    return candidates


def compare_16bit_candidates(candidates):
    le = candidates["s16le"]["stats"]
    be = candidates["s16be"]["stats"]

    print("\n=== 16-BIT ENDIANNESS COMPARISON ===")
    print(
        "The heuristic below compares waveform continuity. "
        "Real audio usually changes relatively smoothly between adjacent samples, "
        "while the wrong byte order often creates much larger jumps."
    )

    print(
        f"s16le smoothness ratio: {le['smoothness_ratio']:.5f}"
    )
    print(
        f"s16be smoothness ratio: {be['smoothness_ratio']:.5f}"
    )

    if math.isfinite(le["smoothness_ratio"]) and math.isfinite(be["smoothness_ratio"]):
        if le["smoothness_ratio"] < be["smoothness_ratio"] * 0.75:
            print(
                "\nHeuristic result: LITTLE-ENDIAN looks substantially more "
                "audio-like based on adjacent-sample continuity."
            )
        elif be["smoothness_ratio"] < le["smoothness_ratio"] * 0.75:
            print(
                "\nHeuristic result: BIG-ENDIAN looks substantially more "
                "audio-like based on adjacent-sample continuity."
            )
        else:
            print(
                "\nHeuristic result: continuity alone is not decisive. "
                "Listen to both WAV files and inspect their waveforms/spectrograms."
            )

    print("\nThis heuristic is evidence, not proof. Listening is the final sanity check.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a possible raw audio file and optionally create "
            "16-bit little-endian / big-endian WAV candidates."
        )
    )

    parser.add_argument(
        "path",
        help="Path to the .raw file"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Known/expected duration in seconds, e.g. 67.3"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
        help="Expected sample rate. Default: 8000 Hz"
    )

    parser.add_argument(
        "--no-wav-candidates",
        action="store_true",
        help="Do not create s16le/s16be candidate WAV files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for candidate WAV files. "
            "Default: <raw filename>_candidates next to the raw file"
        )
    )

    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()

    if not path.exists():
        raise SystemExit(f"File does not exist: {path}")

    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    size_bytes = path.stat().st_size

    print("=" * 70)
    print("RAW AUDIO INSPECTION")
    print("=" * 70)

    print(f"\nFile:        {path}")
    print(f"Size:        {size_bytes:,} bytes ({human_size(size_bytes)})")
    print(f"Assumed SR:  {args.sample_rate} Hz")

    if args.duration is not None:
        print(f"Known length: {args.duration:.3f} seconds")

        bytes_per_time_sample = (
            size_bytes /
            (args.sample_rate * args.duration)
        )

        print(
            "Observed bytes per time-sample: "
            f"{bytes_per_time_sample:.4f}"
        )

    # ------------------------------------------------------------
    # Standard file identification
    # ------------------------------------------------------------

    print("\n=== FILE COMMAND ===")
    print(inspect_with_file(path))

    ffprobe_data, ffprobe_error = inspect_with_ffprobe(path)

    if ffprobe_data:
        print_ffprobe(ffprobe_data)
    else:
        print("\n=== FFPROBE ===")
        print(ffprobe_error)

    # ------------------------------------------------------------
    # Raw PCM hypotheses
    # ------------------------------------------------------------

    print("\n=== POSSIBLE RAW ENCODINGS ===")

    results = analyze_candidates(
        size_bytes,
        args.sample_rate,
        args.duration
    )

    if args.duration is None:
        print(
            "\nNo reference duration supplied, so these are only "
            "the durations each encoding would imply.\n"
        )

        for result in results:
            print(
                f"{result['name']:<48} "
                f"-> {result['implied_duration']:8.3f} s"
            )

    else:
        print(
            "\nRanked by agreement with the expected duration:\n"
        )

        for i, result in enumerate(results[:10], start=1):
            print(
                f"{i:2}. {result['name']:<46} "
                f"duration={result['implied_duration']:8.3f} s  "
                f"error={result['difference']:7.3f} s "
                f"({100 * result['relative_error']:.3f}%)"
            )

        best = results[0]

        print("\n=== BEST SIZE-BASED MATCH ===")
        print(best["name"])
        print(
            f"Implied duration: {best['implied_duration']:.3f} seconds"
        )
        print(
            f"Difference from expected duration: "
            f"{best['difference']:.3f} seconds"
        )

    # ------------------------------------------------------------
    # Create 16-bit endian candidates
    # ------------------------------------------------------------

    if not args.no_wav_candidates:
        if output_dir_arg := args.output_dir:
            output_dir = Path(output_dir_arg).expanduser().resolve()
        else:
            output_dir = path.parent / f"{path.stem}_candidates"

        print("\n=== CREATING 16-BIT MONO WAV CANDIDATES ===")
        print(f"Output directory: {output_dir}")

        candidates = create_16bit_candidates(
            path,
            args.sample_rate,
            output_dir
        )

        for suffix in ("s16le", "s16be"):
            item = candidates[suffix]
            print_statistics(
                f"{suffix.upper()} ({item['endian']}-endian input)",
                item["stats"]
            )
            print(f"WAV written to:         {item['wav_path']}")

        compare_16bit_candidates(candidates)

    print("\n=== IMPORTANT ===")
    print(
        "File size can strongly identify bit depth/channel count, "
        "but it cannot by itself prove byte order (endianness)."
    )
    print(
        "The candidate WAV files let you verify byte order by listening "
        "and by comparing waveform statistics."
    )


if __name__ == "__main__":
    main()

# python "/Users/akimdavydov/docs/PW/inspect_raw.py" "/Users/akimdavydov/docs/PW/data/haydn_sonata24_mv1.000.raw" --duration 102