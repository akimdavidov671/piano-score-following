from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "score-notes"
DEFAULT_OUTPUT = SCRIPT_DIR / "score-states"
SUMMARY_NAME = "extraction_summary.csv"
# Diagnostic only: exceptionally short intervals, not corrected or removed.
SHORT_LENGTH = Fraction(1, 32)  # one thirty-second of a quarter note

STATE_FIELDS = [
    "state_index", "start_quarter", "end_quarter", "length_quarter",
    "start_whole_fraction", "end_whole_fraction", "length_whole_fraction",
    "active_midi_pitches", "active_pitch_count", "active_note_count",
    "note_on_count", "note_off_count", "is_rest",
]
STATE_DTYPE = np.dtype([
    ("state_index", "i4"),
    ("start_quarter", "f8"), ("end_quarter", "f8"),
    ("length_quarter", "f8"),
    ("start_whole_fraction", "U64"),
    ("end_whole_fraction", "U64"),
    ("length_whole_fraction", "U64"),
    ("active_midi_pitches", "U1024"),
    ("active_pitch_count", "i4"), ("active_note_count", "i4"),
    ("note_on_count", "i4"), ("note_off_count", "i4"),
    ("is_rest", "?"),
])
SUMMARY_FIELDS = [
    "source_file", "variant", "status", "ordinary_notes", "grace_excluded",
    "grace_with_nonzero_duration_excluded", "state_count", "rest_states",
    "short_states_under_1_32_quarter", "consecutive_identical_pitch_sets",
    "first_start_quarter", "last_end_quarter", "csv", "npz", "error",
]


def _is_true(value: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Unexpected is_grace value: {value!r}")


def _quarter_from_whole(raw: str, label: str) -> Fraction:
    if not raw.strip():
        raise ValueError(f"Missing {label} fraction")
    try:
        value = Fraction(raw.strip()) * 4
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid {label} fraction: {raw!r}") from exc
    return value


def _check_float_column(raw: str, exact_q: Fraction, field: str, row_num: int):
    """Check source float agrees with exported rational position, allowing
    only the small representation error from the earlier exporter."""
    try:
        reported = float(raw)
    except ValueError as exc:
        raise ValueError(f"Row {row_num}: invalid {field}: {raw!r}") from exc
    if not math.isfinite(reported) or not math.isclose(reported, float(exact_q), rel_tol=0, abs_tol=2e-6):
        raise ValueError(
            f"Row {row_num}: inconsistent {field}={raw!r} and rational position {exact_q}"
        )


def read_notes(path: Path, expected_file: str, expected_variant: str = "original"):
    """Use precise note positions already exported in whole-note fractions.

    Ordinary notes drive the state timeline; grace notes are counted but not
    placed in that timeline. The source CSV is read-only.
    """
    ordinary = []
    grace_total = grace_positive = 0
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "source_file", "timing_variant", "pitch_midi", "is_grace",
            "onset_quarter", "offset_quarter",
            "onset_whole_fraction", "offset_whole_fraction",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            if row["source_file"] != expected_file:
                raise ValueError(f"Row {row_num}: unexpected source_file {row['source_file']!r}")
            if row["timing_variant"] != expected_variant:
                raise ValueError(
                    f"Row {row_num}: timing variant {row['timing_variant']!r} "
                    f"does not match expected {expected_variant!r}"
                )
            onset = _quarter_from_whole(row["onset_whole_fraction"], "onset")
            offset = _quarter_from_whole(row["offset_whole_fraction"], "offset")
            _check_float_column(row["onset_quarter"], onset, "onset_quarter", row_num)
            _check_float_column(row["offset_quarter"], offset, "offset_quarter", row_num)
            if offset < onset:
                raise ValueError(f"Row {row_num}: note offset before onset")
            pitch = int(row["pitch_midi"])
            if pitch < 0 or pitch > 127:
                raise ValueError(f"Row {row_num}: invalid MIDI pitch {pitch}")
            if _is_true(row["is_grace"]):
                grace_total += 1
                if offset > onset:
                    grace_positive += 1
                continue
            if offset == onset:
                raise ValueError(f"Row {row_num}: ordinary note has zero duration")
            ordinary.append((pitch, onset, offset))
    return ordinary, grace_total, grace_positive


def build_states(notes: list[tuple[int, Fraction, Fraction]]):
    """Sweep through on/off events in rational score time.

    At a boundary, process note-offs and note-ons together, then record the
    resulting active notes over the following half-open interval [t_i,t_{i+1}).
    A Counter (not a set) is required because two voices may play the same
    pitch simultaneously, with different durations.
    """
    if not notes:
        raise ValueError("No ordinary notes available to construct score states")
    starts: dict[Fraction, Counter] = defaultdict(Counter)
    ends: dict[Fraction, Counter] = defaultdict(Counter)
    for pitch, begin, end in notes:
        starts[begin][pitch] += 1
        ends[end][pitch] += 1
    boundaries = set(starts) | set(ends)
    first_onset = min(starts)
    if first_onset > 0:
        boundaries.add(Fraction(0))  # initial rest / delayed first entrance
    sorted_boundaries = sorted(boundaries)
    active: Counter = Counter()
    states = []
    for i, start in enumerate(sorted_boundaries[:-1]):
        # Notes ending at this boundary are inactive in the NEXT interval.
        for pitch, count in ends[start].items():
            active[pitch] -= count
            if active[pitch] < 0:
                raise ValueError(f"Negative active-note count for pitch {pitch} at {start}")
            if active[pitch] == 0:
                del active[pitch]
        for pitch, count in starts[start].items():
            active[pitch] += count
        end = sorted_boundaries[i + 1]
        if end <= start:
            raise ValueError(f"Non-increasing boundary sequence at {start}")
        pitches = sorted(active)
        states.append({
            "state_index": i + 1,    # 1-based, matching Jiang's Table 2.1
            "start_quarter": float(start),
            "end_quarter": float(end),
            "length_quarter": float(end - start),
            "start_whole_fraction": str(start / 4),
            "end_whole_fraction": str(end / 4),
            "length_whole_fraction": str((end - start) / 4),
            "active_midi_pitches": json.dumps(pitches, separators=(",", ":")),
            "active_pitch_count": len(pitches),
            "active_note_count": sum(active.values()),
            "note_on_count": sum(starts[start].values()),
            "note_off_count": sum(ends[start].values()),
            "is_rest": not bool(active),
        })
    # The final boundary has no following interval and does not produce a state.
    final = sorted_boundaries[-1]
    for pitch, count in ends[final].items():
        active[pitch] -= count
        if active[pitch] < 0:
            raise ValueError(f"Negative active-note count for pitch {pitch} at final boundary {final}")
        if active[pitch] == 0:
            del active[pitch]
    for pitch, count in starts[final].items():
        active[pitch] += count
    if active:
        raise ValueError(f"Notes still active after final boundary: {dict(active)}")
    if len(states) != len(sorted_boundaries) - 1:
        raise AssertionError("Missing state interval")
    return states


def save_states(states: list[dict], stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Do not use Path.with_suffix: some score stems themselves contain dots.
    path_csv = Path(str(stem) + ".csv")
    path_npz = Path(str(stem) + ".npz")
    with path_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STATE_FIELDS)
        writer.writeheader()
        writer.writerows(states)
    data = np.empty(len(states), dtype=STATE_DTYPE)
    for i, row in enumerate(states):
        data[i] = tuple(row[field] for field in STATE_FIELDS)
    np.savez_compressed(path_npz, states=data)
    return path_csv, path_npz


def process_one(info: dict, input_dir: Path, output_dir: Path):
    source = info["source_file"]
    variant = info["variant"]
    output = {key: "" for key in SUMMARY_FIELDS}
    output.update(source_file=source, variant=variant, status="ERROR")
    try:
        if info["status"] != "OK":
            raise ValueError(f"Source extraction summary status is {info['status']}")
        csv_name = Path(info["csv"]).name
        if not csv_name or not csv_name.endswith(".csv"):
            raise ValueError("Missing CSV path in extraction summary")
        input_csv = input_dir / csv_name
        if not input_csv.is_file():
            raise FileNotFoundError(f"Note export not found: {input_csv}")
        ordinary, grace, grace_positive = read_notes(input_csv, source, variant)
        if len(ordinary) != int(info["notes"]) or grace != int(info["grace_notes"]):
            raise ValueError("Note count differs from extraction_summary.csv")
        states = build_states(ordinary)
        output_csv, output_npz = save_states(states, output_dir / Path(csv_name).stem)
        rests = sum(s["is_rest"] for s in states)
        short = sum(Fraction(s["length_whole_fraction"]) * 4 < SHORT_LENGTH for s in states)
        same = sum(
            a["active_midi_pitches"] == b["active_midi_pitches"]
            for a, b in zip(states, states[1:])
        )
        output.update(
            status="OK", ordinary_notes=len(ordinary), grace_excluded=grace,
            grace_with_nonzero_duration_excluded=grace_positive,
            state_count=len(states), rest_states=rests,
            short_states_under_1_32_quarter=short,
            consecutive_identical_pitch_sets=same,
            first_start_quarter=states[0]["start_quarter"],
            last_end_quarter=states[-1]["end_quarter"],
            csv=str(output_csv), npz=str(output_npz), error="",
        )
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


COMPARISON_FIELDS = [
    "source_file", "adjusted_variant", "status", "ordinary_notes",
    "original_states", "adjusted_states", "states_removed", "state_reduction_pct",
    "original_rests", "adjusted_rests", "rests_removed",
    "original_short_states", "adjusted_short_states", "short_states_removed",
    "original_identical_pitch_sets", "adjusted_identical_pitch_sets",
    "timing_report_check", "error",
]


def load_summary(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty summary: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _validate_unique(rows: list[dict], label: str):
    keys = [(r["source_file"], r["variant"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate source/variant entries in {label}")


def run_batch(*, notes_dir: Path, states_dir: Path,
              expected: str, file_filter: str | None = None) -> list[dict]:
    """Process one isolated variant family; never co-mingle output directories."""
    source_summary = notes_dir / SUMMARY_NAME
    entries = load_summary(source_summary)
    if expected == "original":
        entries = [r for r in entries if r["variant"] == "original"]
    else:
        entries = [r for r in entries if r["variant"] != "original"]
    if file_filter:
        entries = [r for r in entries if r["source_file"] == file_filter]
    if not entries:
        raise ValueError(f"No {expected} note exports found in {source_summary}"
                         + (f" for {file_filter!r}" if file_filter else ""))
    _validate_unique(entries, str(source_summary))
    if expected == "adjusted" and len({r["source_file"] for r in entries}) != len(entries):
        raise ValueError("Multiple adjusted variants of one source in the same directory; "
                         "use separate directories to avoid ambiguous comparisons")
    print(f"\nBuilding {len(entries)} {expected} score-state sequence(s)")
    print(f"  Notes:  {notes_dir}\n  States: {states_dir}")
    results = []
    for entry in entries:
        result = process_one(entry, notes_dir, states_dir)
        results.append(result)
        print(f"  [{result['status']}] {entry['source_file']} ({entry['variant']}): "
              f"{result['state_count']} states, {result['rest_states']} rests"
              + (f" — {result['error']}" if result["error"] else ""))
    _write_csv(states_dir / "score_state_summary.csv", results, SUMMARY_FIELDS)
    if any(r["status"] != "OK" for r in results):
        raise ValueError(f"{sum(r['status']!='OK' for r in results)} failed "
                         f"{expected} score-state construction(s); see {states_dir/'score_state_summary.csv'}")
    return results


def compare_variants(original: list[dict], adjusted: list[dict],
                     timing_report: Path | None = None) -> list[dict]:
    """Compare actual state-builder outputs; optionally verify transformation diagnostics."""
    base = {r["source_file"]: r for r in original if r["variant"] == "original"}
    if len(base) != sum(r["variant"] == "original" for r in original):
        raise ValueError("Duplicate original source in baseline state summary")
    report = {}
    if timing_report is not None and timing_report.is_file():
        report_rows = load_summary(timing_report)
        report = {(r["source_file"], r["variant"]): r for r in report_rows}
    comparisons = []
    seen_sources = set()
    for a in adjusted:
        source = a["source_file"]
        variant = a["variant"]
        if source in seen_sources:
            raise ValueError(f"Duplicate adjusted source in comparison: {source}")
        seen_sources.add(source)
        if source not in base:
            raise ValueError(f"No original score-state summary for adjusted source: {source}")
        b = base[source]
        if a["status"] != "OK" or b["status"] != "OK":
            raise ValueError(f"Cannot compare non-OK score-state records: {source}")
        if int(b["ordinary_notes"]) != int(a["ordinary_notes"]):
            raise ValueError(f"Note count changed between versions: {source}")
        old_states, new_states = int(b["state_count"]), int(a["state_count"])
        old_rests, new_rests = int(b["rest_states"]), int(a["rest_states"])
        old_short = int(b["short_states_under_1_32_quarter"])
        new_short = int(a["short_states_under_1_32_quarter"])
        check = "NOT_AVAILABLE"
        if (source, variant) in report:
            ref = report[(source, variant)]
            fields = {
                "original_states": old_states, "adjusted_states": new_states,
                "original_rests": old_rests, "adjusted_rests": new_rests,
                "original_short_states": old_short, "adjusted_short_states": new_short,
            }
            for key, actual in fields.items():
                if ref.get(key, "") and int(ref[key]) != actual:
                    raise ValueError(f"{source}: state builder {key}={actual}, "
                                     f"but timing report says {ref[key]}")
            check = "MATCH"
        comparisons.append({
            "source_file": source, "adjusted_variant": variant,
            "status": "OK", "ordinary_notes": b["ordinary_notes"],
            "original_states": old_states, "adjusted_states": new_states,
            "states_removed": old_states-new_states,
            "state_reduction_pct": round(100*(old_states-new_states)/old_states, 3),
            "original_rests": old_rests, "adjusted_rests": new_rests,
            "rests_removed": old_rests-new_rests,
            "original_short_states": old_short,
            "adjusted_short_states": new_short,
            "short_states_removed": old_short-new_short,
            "original_identical_pitch_sets": b["consecutive_identical_pitch_sets"],
            "adjusted_identical_pitch_sets": a["consecutive_identical_pitch_sets"],
            "timing_report_check": check,
            "error": "",
        })
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Root location of Step 2.2 note exports")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Root destination for score-state CSV/NPZ files")
    parser.add_argument("--variant", choices=("original", "adjusted", "both"),
                        default="original",
                        help="original (default), adjusted (4 variants), or both (20 states)")
    parser.add_argument("--file", help="Limit processing to one original source filename")
    args = parser.parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        original_rows = []
        adjusted_rows = []
        if args.variant in ("original", "both"):
            original_rows = run_batch(notes_dir=input_dir, states_dir=output_dir,
                                      expected="original", file_filter=args.file)
        if args.variant in ("adjusted", "both"):
            adjusted_rows = run_batch(notes_dir=input_dir / "timing-adjusted",
                                      states_dir=output_dir / "timing-adjusted",
                                      expected="adjusted", file_filter=args.file)
            if args.variant == "adjusted":
                original_summary = output_dir / "score_state_summary.csv"
                if not original_summary.is_file():
                    raise FileNotFoundError(
                        f"Missing original state summary {original_summary}. "
                        "Run the script with --variant original first.")
                original_rows = load_summary(original_summary)
            comparisons = compare_variants(
                original_rows, adjusted_rows,
                timing_report=input_dir / "timing-adjusted" / "timing_correction_summary.csv",
            )
            report = output_dir / "score_state_comparison.csv"
            _write_csv(report, comparisons, COMPARISON_FIELDS)
            print(f"\nCompared {len(comparisons)} adjusted MIDI variants with originals:")
            for row in comparisons:
                print(f"  {row['source_file']}: {row['original_states']} -> "
                      f"{row['adjusted_states']} states; "
                      f"{row['original_rests']} -> {row['adjusted_rests']} rests; "
                      f"{row['original_short_states']} -> {row['adjusted_short_states']} short")
            print(f"Comparison: {report}")
        print("\nDone. All original note records remain unmodified.")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
