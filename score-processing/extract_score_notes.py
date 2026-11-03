from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SCORE_DIR = PROJECT_DIR / "score"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "score-notes"
DEFAULT_CONFIG = SCRIPT_DIR / "score_part_selection.json"
SUFFIXES = {".mxl", ".xml", ".musicxml", ".mid", ".midi"}

FIELDS = [
    "source_file", "timing_variant", "part_index", "part_name", "note_id",
    "pitch_midi", "onset_quarter", "offset_quarter", "duration_quarter",
    "onset_whole_fraction", "offset_whole_fraction", "duration_whole_fraction",
    "onset_div", "offset_div", "measure_number", "measure_onset_quarter",
    "staff", "voice", "is_grace", "grace_type", "tied_segments",
]
# Fixed-width strings allow np.load(..., allow_pickle=False), including for empty arrays.
DTYPE = np.dtype([
    ("source_file", "U256"), ("timing_variant", "U32"),
    ("part_index", "i4"), ("part_name", "U128"), ("note_id", "U128"),
    ("pitch_midi", "i4"), ("onset_quarter", "f8"),
    ("offset_quarter", "f8"), ("duration_quarter", "f8"),
    ("onset_whole_fraction", "U32"), ("offset_whole_fraction", "U32"),
    ("duration_whole_fraction", "U32"), ("onset_div", "i8"),
    ("offset_div", "i8"), ("measure_number", "U32"),
    ("measure_onset_quarter", "f8"), ("staff", "U32"),
    ("voice", "U32"), ("is_grace", "?"), ("grace_type", "U48"),
    ("tied_segments", "i4"),
])


def whole_fraction(quarter: float) -> str:
    """Human-readable fraction of a whole note (quarter note = 1/4)."""
    value = Fraction(float(quarter)).limit_denominator(1_000_000) / 4
    return str(value)


def plain(value: Any) -> str:
    return "" if value is None else str(value)


def measure_at(measures: list, measure_starts: list[int], onset_div: int):
    """Return the containing measure, preferring the next at bar boundaries."""
    i = bisect_right(measure_starts, onset_div) - 1
    if i < 0:
        return None
    m = measures[i]
    if int(m.start.t) <= onset_div < int(m.end.t):
        return m
    return None


def tied_length(note) -> int:
    """Count noteheads in a tie group; catch unexpectedly cyclic ties."""
    count, seen, current = 0, set(), note
    while current is not None:
        if id(current) in seen:
            raise ValueError("Cyclic tied-note reference in Partitura score")
        seen.add(id(current))
        count += 1
        current = getattr(current, "tie_next", None)
    return count


def extract_part(part, part_index: int, source_file: str, variant: str, pt):
    """Extract one event per tied group; preserve grace-note information."""
    rows = []
    qmap = part.quarter_map
    measures = sorted(
        (m for m in getattr(part, "measures", []) if m.start is not None and m.end is not None),
        key=lambda m: (int(m.start.t), int(m.end.t)),
    )
    measure_starts = [int(m.start.t) for m in measures]
    for note in part.notes_tied:
        if note.start is None or note.end is None:
            raise ValueError(f"Note missing start/end in {source_file}, P{part_index}")
        is_grace = isinstance(note, pt.score.GraceNote)
        start = int(note.start.t)
        # end_tied follows the tie to its final endpoint, unlike note.end.
        end_point = getattr(note, "end_tied", None)
        end = int(end_point.t if end_point is not None else note.end.t)
        onset_q, offset_q = float(qmap(start)), float(qmap(end))
        if end < start or offset_q < onset_q - 1e-9:
            raise ValueError(f"Negative note duration: {source_file}, P{part_index}, {note.id}")
        if not is_grace and end == start:
            raise ValueError(f"Zero-duration ordinary note: {source_file}, P{part_index}, {note.id}")
        pitch = int(note.midi_pitch)
        if not 0 <= pitch <= 127:
            raise ValueError(f"Invalid MIDI pitch {pitch}: {source_file}, P{part_index}")
        measure = measure_at(measures, measure_starts, start)
        measure_q = float(qmap(int(measure.start.t))) if measure else float("nan")
        rows.append({
            "source_file": source_file,
            "timing_variant": variant,
            "part_index": part_index,
            "part_name": plain(getattr(part, "part_name", None)),
            "note_id": plain(getattr(note, "id", None)),
            "pitch_midi": pitch,
            "onset_quarter": onset_q,
            "offset_quarter": offset_q,
            "duration_quarter": offset_q - onset_q,
            "onset_whole_fraction": whole_fraction(onset_q),
            "offset_whole_fraction": whole_fraction(offset_q),
            "duration_whole_fraction": whole_fraction(offset_q - onset_q),
            "onset_div": start,
            "offset_div": end,
            "measure_number": plain(getattr(measure, "number", None)),
            "measure_onset_quarter": measure_q,
            "staff": plain(getattr(note, "staff", None)),
            "voice": plain(getattr(note, "voice", None)),
            "is_grace": is_grace,
            "grace_type": plain(getattr(note, "grace_type", None)) if is_grace else "",
            "tied_segments": tied_length(note),
        })
    return rows


def sorted_rows(rows: list[dict]):
    # Stable order: score onset first, then grace flag and pitch/part.
    return sorted(rows, key=lambda r: (
        r["onset_quarter"], r["is_grace"], r["pitch_midi"],
        r["part_index"], r["offset_quarter"], r["note_id"],
    ))


def write_outputs(rows: list[dict], dest_stem: Path):
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(dest_stem) + ".csv")
    npz_path = Path(str(dest_stem) + ".npz")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    array = np.empty(len(rows), dtype=DTYPE)
    for i, row in enumerate(rows):
        array[i] = tuple(row[field] for field in FIELDS)
    np.savez_compressed(npz_path, notes=array)
    return csv_path, npz_path


def get_parts(score: Any):
    """Handle current Score and older list/Part loader return types."""
    if hasattr(score, "parts"):
        return list(score.parts)
    if hasattr(score, "notes_tied"):
        return [score]
    if isinstance(score, (list, tuple)):
        parts = []
        for obj in score:
            parts.extend(get_parts(obj))
        return parts
    raise TypeError(f"Partitura returned unsupported object type: {type(score)}")


def get_selection(filename: str, parts: list, config: dict):
    data = config.get(filename)
    if data is None:
        if len(parts) == 1:
            return [1], "automatic: one part", "confirmed"
        raise ValueError(
            f"{filename} contains {len(parts)} parts but has no explicit "
            "selection in score_part_selection.json"
        )
    selected = data.get("parts")
    if not isinstance(selected, list) or not selected or any(
        not isinstance(x, int) or isinstance(x, bool) for x in selected
    ):
        raise ValueError(f"Invalid part selection for {filename}: {selected}")
    if len(set(selected)) != len(selected) or min(selected) < 1 or max(selected) > len(parts):
        raise ValueError(f"Out-of-range/duplicate part selection for {filename}: {selected}")
    return selected, plain(data.get("reason", "explicit selection")), plain(data.get("status", "review"))


def process_one(path: Path, config: dict, output_dir: Path, pt, *, strict: bool = False):
    """Export only original score information; no timing transformations."""
    try:
        if path.suffix.lower() in {".mid", ".midi"}:
            # Preserve source MIDI events on the finest available (one-tick) grid.
            score = pt.load_score_midi(str(path), quantization_unit=1)
        else:
            score = pt.load_score(str(path))
        parts = get_parts(score)
        selected, reason, status = get_selection(path.name, parts, config)
        rows = []
        for ix in selected:
            rows.extend(extract_part(parts[ix - 1], ix, path.name, "original", pt))
        rows = sorted_rows(rows)
        ordinary = sum(not r["is_grace"] for r in rows)
        grace = len(rows) - ordinary
        if ordinary == 0:
            raise ValueError("Selected parts contain zero ordinary notes")
        safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem)[:130]
        csv_out, npz_out = write_outputs(rows, output_dir / safe_base)
        result = {"source_file": path.name, "variant": "original", "status": "OK",
                  "selected_parts": ";".join(str(i) for i in selected),
                  "selection_status": status, "selection_reason": reason,
                  "notes": ordinary, "grace_notes": grace,
                  "tied_groups": sum(r["tied_segments"] > 1 for r in rows),
                  "missing_measure": sum(not r["measure_number"] for r in rows),
                  "missing_staff": sum(not r["staff"] for r in rows),
                  "csv": str(csv_out), "npz": str(npz_out), "error": ""}
    except Exception as exc:
        if strict:
            raise
        result = {"source_file": path.name, "variant": "original", "status": "ERROR",
                  "selected_parts": "", "selection_status": "", "selection_reason": "",
                  "notes": "", "grace_notes": "", "tied_groups": "",
                  "missing_measure": "", "missing_staff": "", "csv": "", "npz": "",
                  "error": f"{type(exc).__name__}: {exc}"}
    print(f"  [{result['status']}] {path.name} (original): "
          f"{result['notes']} notes, {result['grace_notes']} grace"
          + (f" — {result['error']}" if result["error"] else ""))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR,
                    help="Input score directory (default: PW/score)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Output directory (default: score-processing/score-notes)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="JSON part-selection configuration")
    ap.add_argument("--file", help="Inspect only one score filename within --score-dir")
    ap.add_argument("--strict", action="store_true", help="Stop at the first error")
    args = ap.parse_args(argv)
    try:
        import partitura as pt
    except ImportError as exc:
        ap.error("Partitura is missing. Install it with: python -m pip install partitura")
    score_dir = args.score_dir.resolve()
    out_dir = args.output_dir.resolve()
    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        ap.error("Part-selection config must be a JSON object")
    if not score_dir.is_dir():
        ap.error(f"Score directory not found: {score_dir}")
    if args.file:
        if Path(args.file).name != args.file:
            ap.error("--file must be a filename, not a directory path")
        paths = [score_dir / args.file]
        if not paths[0].is_file():
            ap.error(f"Score file not found: {paths[0]}")
    else:
        paths = sorted(p for p in score_dir.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not paths:
        ap.error(f"No supported scores in {score_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {len(paths)} scores from {score_dir}")
    print(f"Writing note-level records to {out_dir}\n")
    results = []
    for path in paths:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            current = process_one(path, config, out_dir, pt, strict=args.strict)
        current["parser_warning_count"] = len(caught)
        results.append(current)
    report = out_dir / "extraction_summary.csv"
    with report.open("w", encoding="utf-8", newline="") as f:
        fields = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    failures = sum(r["status"] != "OK" for r in results)
    print(f"\nSummary: {len(results) - failures} exported, {failures} failed.")
    print(f"Summary report: {report}")
    if failures:
        print("Review ERROR rows in extraction_summary.csv; other scores were still exported.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
