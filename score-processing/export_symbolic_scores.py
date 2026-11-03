from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATES = SCRIPT_DIR / "score-states"
DEFAULT_OUTPUT = SCRIPT_DIR / "symbolic-scores"

# The Step 2.3 exports may use slightly different pitch-column spellings.
PITCH_COLUMNS = (
    "active_midi_pitches", "active_pitches_midi", "pitches_midi",
    "active_midi_pitches_json", "active_pitches_midi_json", "pitches_midi_json",
    "pitch_set_midi", "midi_pitch_set", "active_pitch_set",
    "active_pitches", "active_pitches_json", "pitch_set_json", "pitch_set",
    "pitches", "pitches_json", "chord_pitches", "midi_pitches",
)
INDEX_COLUMNS = ("state_index", "index", "chord_index")
START_WHOLE_COLUMNS = (
    "start_whole_fraction", "onset_whole_fraction", "score_position_whole_fraction",
    "start_whole", "onset_whole", "score_position_whole",
)
LENGTH_WHOLE_COLUMNS = (
    "length_whole_fraction", "duration_whole_fraction", "nominal_length_whole_fraction",
    "length_whole", "duration_whole", "nominal_length_whole",
)
START_QUARTER_COLUMNS = ("start_quarter", "onset_quarter", "score_position_quarter")
LENGTH_QUARTER_COLUMNS = ("length_quarter", "duration_quarter", "nominal_length_quarter")

COLUMNS = (
    "state_index", "score_position_whole", "nominal_length_whole",
    "active_midi_pitches", "is_rest",
)
SUMMARY_COLUMNS = (
    "source_file", "variant", "status", "state_count", "rest_states",
    "pitch_column_in_source", "input_csv", "output_csv", "error",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no column headers: {path}")
        return list(reader.fieldnames), list(reader)


def field_name(fields: list[str], choices: tuple[str, ...], name: str,
               *, optional: bool = False) -> str | None:
    for candidate in choices:
        if candidate in fields:
            return candidate
    if optional:
        return None
    raise ValueError(f"Missing {name} column. Available columns: {', '.join(fields)}")


def parse_fraction(value: str, context: str) -> Fraction:
    try:
        result = Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"Invalid musical position/length {value!r} at {context}") from exc
    return result


def parse_pitches(raw: str, *, context: str) -> list[int]:
    text = str(raw).strip()
    if not text or text.lower() in ("none", "set()"):
        return []
    # Accept JSON arrays, Python tuples/sets, or delimiters such as '60;64;67'.
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if not re.fullmatch(r"[0-9\s,;|\[\](){}]+", text):
                raise ValueError(f"Unrecognized active-pitch encoding {text!r} at {context}")
            parsed = re.findall(r"\d+", text)
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError(f"Expected a list/set of MIDI pitches, got {text!r} at {context}")
    try:
        values = [int(p) for p in parsed]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Non-integer pitch at {context}: {text!r}") from exc
    if any(not 0 <= p <= 127 for p in values):
        raise ValueError(f"MIDI pitch outside 0..127 at {context}: {text!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate pitch in an active-pitch SET at {context}: {text!r}")
    return sorted(values)


def parse_bool(raw: str, context: str) -> bool:
    value = str(raw).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid rest indicator {raw!r} at {context}")


def export_one(source_csv: Path, destination_csv: Path,
               *, source_file: str, variant: str, expected_states: int):
    columns, rows = read_csv(source_csv)
    index_col = field_name(columns, INDEX_COLUMNS, "state index")
    start_w_col = field_name(columns, START_WHOLE_COLUMNS, "whole-note score position", optional=True)
    length_w_col = field_name(columns, LENGTH_WHOLE_COLUMNS, "whole-note musical length", optional=True)
    start_q_col = field_name(columns, START_QUARTER_COLUMNS, "quarter-note score position", optional=True)
    length_q_col = field_name(columns, LENGTH_QUARTER_COLUMNS, "quarter-note musical length", optional=True)
    if not start_w_col and not start_q_col:
        raise ValueError("No score-position column found: " + ", ".join(columns))
    if not length_w_col and not length_q_col:
        raise ValueError("No musical-length column found: " + ", ".join(columns))
    pitch_col = field_name(columns, PITCH_COLUMNS, "active MIDI pitches", optional=True)
    if pitch_col is None:
        # Do not accidentally select note_on_count, note_off_count, or pitch_count.
        choices = [c for c in columns if "pitch" in c.lower()
                   and not any(x in c.lower() for x in ("count", "note_on", "note_off"))]
        if len(choices) != 1:
            raise ValueError("Cannot uniquely identify pitch-set column. "
                             f"Candidates: {choices}; all columns: {columns}")
        pitch_col = choices[0]
    if len(rows) != expected_states:
        raise ValueError(f"Expected {expected_states} states, found {len(rows)} in {source_csv}")

    prev_end = None
    output = []
    rests = 0
    for row_num, row in enumerate(rows, start=1):
        ctx = f"{source_csv.name} row {row_num}"
        index = int(row[index_col])
        if index != row_num:
            raise ValueError(f"Expected consecutive state index {row_num}, got {index} in {ctx}")
        if "source_file" in columns and row["source_file"] != source_file:
            raise ValueError(f"Unexpected source_file in {ctx}")
        if "variant" in columns and row["variant"] != variant:
            raise ValueError(f"Unexpected variant in {ctx}")
        start = parse_fraction(row[start_w_col], ctx) if start_w_col else parse_fraction(row[start_q_col], ctx) / 4
        length = parse_fraction(row[length_w_col], ctx) if length_w_col else parse_fraction(row[length_q_col], ctx) / 4
        if length <= 0:
            raise ValueError(f"Non-positive musical length in {ctx}: {length}")
        if prev_end is not None and start != prev_end:
            raise ValueError(f"Non-contiguous states in {ctx}: previous end={prev_end}, next start={start}")
        if start_q_col and abs(float(start*4) - float(row[start_q_col])) > 2e-6:
            raise ValueError(f"Quarter/whole score position discrepancy in {ctx}")
        if length_q_col and abs(float(length*4) - float(row[length_q_col])) > 2e-6:
            raise ValueError(f"Quarter/whole musical length discrepancy in {ctx}")
        pitches = parse_pitches(row[pitch_col], context=ctx)
        rest = not pitches
        if "is_rest" in columns and parse_bool(row["is_rest"], ctx) != rest:
            raise ValueError(f"Rest indicator disagrees with active pitches in {ctx}")
        output.append({
            "state_index": index,
            "score_position_whole": str(start),
            "nominal_length_whole": str(length),
            "active_midi_pitches": json.dumps(pitches, separators=(",", ":")),
            "is_rest": rest,
        })
        rests += rest
        prev_end = start + length

    # Validate ALL files before writing to avoid producing a partial table.
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    with destination_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(output)
    return {"state_count": len(output), "rest_states": rests,
            "pitch_column_in_source": pitch_col}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states-dir", type=Path, default=DEFAULT_STATES,
                        help="Directory containing the Step 2.3 state exports")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="New destination for final symbolic score tables")
    args = parser.parse_args(argv)
    states_dir = args.states_dir.resolve()
    output_dir = args.output_dir.resolve()
    if states_dir == output_dir or states_dir in output_dir.parents or output_dir in states_dir.parents:
        parser.error("Output must not be inside or equal to the source states directory")

    # Reuse the established manifests; do not rediscover/re-parse source scores.
    manifests = (
        (states_dir / "score_state_summary.csv", states_dir, "original"),
        (states_dir / "timing-adjusted" / "score_state_summary.csv",
         states_dir / "timing-adjusted", "timing-adjusted"),
    )
    manifest_rows = []
    try:
        for manifest_path, input_folder, family in manifests:
            _, entries = read_csv(manifest_path)
            if not entries:
                raise ValueError(f"Empty state summary {manifest_path}")
            for entry in entries:
                if entry["status"] != "OK":
                    raise ValueError(f"Non-OK state summary row: {entry['source_file']}")
                if family == "original" and entry["variant"] != "original":
                    raise ValueError("Unexpected non-original state in baseline manifest")
                if family != "original" and entry["variant"] == "original":
                    raise ValueError("Unexpected original state in adjusted manifest")
                source = entry["source_file"]
                variant = entry["variant"]
                name = Path(entry["csv"]).name  # absolute paths from other machines are ignored
                input_csv = input_folder / name
                output_csv = output_dir / family / name
                manifest_rows.append((source, variant, entry, input_csv, output_csv))
        if len(manifest_rows) != len(set((s, v) for s, v, *_ in manifest_rows)):
            raise ValueError("Duplicate source/variant in state summaries")
        if len(manifest_rows) != 20:
            raise ValueError(f"Expected 16 original + 4 adjusted scores (20 total), "
                             f"found {len(manifest_rows)}. Check your Step 2.3 summaries")

        summary_rows = []
        for source, variant, entry, input_csv, output_csv in manifest_rows:
            counts = export_one(input_csv, output_csv,
                                source_file=source, variant=variant,
                                expected_states=int(entry["state_count"]))
            if counts["rest_states"] != int(entry["rest_states"]):
                output_csv.unlink(missing_ok=True)
                raise ValueError(f"Rest count does not match state manifest: {source} ({variant})")
            summary_rows.append({
                "source_file": source, "variant": variant, "status": "OK",
                **counts,
                "input_csv": str(input_csv), "output_csv": str(output_csv), "error": "",
            })
            print(f"  [OK] {source} ({variant}): {counts['state_count']} states")
        summary_csv = output_dir / "symbolic_score_summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nFinished: {len(summary_rows)} final symbolic score tables")
        print(f"Summary: {summary_csv}")
        print("Original note and score-state files were not modified.")
        return 0
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
