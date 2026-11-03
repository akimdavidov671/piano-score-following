from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter, defaultdict, deque
from fractions import Fraction
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
NOTE_DIR = SCRIPT_DIR / "score-notes"
SCORE_DIR = PROJECT_DIR / "score"
CONFIG = SCRIPT_DIR / "score_part_selection.json"
SUPPORTED = {".mxl", ".xml", ".musicxml", ".mid", ".midi"}

REQUIRED_FIELDS = (
    "source_file", "timing_variant", "part_index", "part_name", "note_id",
    "pitch_midi", "onset_quarter", "offset_quarter", "duration_quarter",
    "onset_whole_fraction", "offset_whole_fraction", "duration_whole_fraction",
    "onset_div", "offset_div", "measure_number", "measure_onset_quarter",
    "staff", "voice", "is_grace", "grace_type", "tied_segments",
)
INT_FIELDS = {"part_index", "pitch_midi", "onset_div", "offset_div", "tied_segments"}
FLOAT_FIELDS = {"onset_quarter", "offset_quarter", "duration_quarter", "measure_onset_quarter"}
BOOL_FIELDS = {"is_grace"}
FRACTION_FIELDS = {"onset_whole_fraction", "offset_whole_fraction", "duration_whole_fraction"}
REPORT_CAP = 12  # keep reports readable even if a file has thousands of bad rows
TIME_TOL = 1e-7


class Checks:
    def __init__(self, source, variant):
        self.source = source
        self.variant = variant
        self.issues = []
        self.counts = Counter()

    def add(self, severity, code, description, *, record=""):
        self.counts[severity] += 1
        self.issues.append({
            "source_file": self.source, "variant": self.variant,
            "severity": severity, "code": code, "record": str(record),
            "description": str(description),
        })

    def capped(self, severity, code, description, *, record=""):
        n = sum(x["code"] == code for x in self.issues)
        if n < REPORT_CAP:
            self.add(severity, code, description, record=record)
        elif n == REPORT_CAP:
            self.add("INFO", code, "Further occurrences omitted from issue report")

    @property
    def status(self):
        if self.counts["FAIL"]:
            return "FAIL"
        if self.counts["REVIEW"]:
            return "REVIEW"
        return "PASS"


def isclose(a, b, tol=TIME_TOL):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def parse_bool(value):
    if value in (True, "True", "true", "1", "1.0", 1):
        return True
    if value in (False, "False", "false", "0", "0.0", 0):
        return False
    raise ValueError(f"Invalid is_grace value: {value!r}")


def typed_csv_rows(path, check):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            missing = set(REQUIRED_FIELDS) - set(columns)
            if missing:
                check.add("FAIL", "CSV_SCHEMA", f"Missing fields: {sorted(missing)}")
                return None
            rows = []
            for n, raw in enumerate(reader, 2):
                try:
                    row = {}
                    for f in REQUIRED_FIELDS:
                        val = raw[f]
                        if val is None:
                            raise ValueError(f"Missing value for {f}")
                        row[f] = (int(val) if f in INT_FIELDS else
                                  float(val) if f in FLOAT_FIELDS else
                                  parse_bool(val) if f in BOOL_FIELDS else val)
                    rows.append(row)
                except (TypeError, ValueError, KeyError) as exc:
                    check.capped("FAIL", "CSV_VALUE", f"CSV line {n}: {exc}", record=n)
            if len(rows) + sum(x["code"] == "CSV_VALUE" for x in check.issues) < 1:
                check.add("FAIL", "EMPTY_CSV", "No usable note rows")
            return rows
    except Exception as exc:
        check.add("FAIL", "CSV_READ", f"Cannot read {path.name}: {type(exc).__name__}: {exc}")
        return None


def read_npz(path, check):
    try:
        with np.load(path, allow_pickle=False) as data:
            if "notes" not in data.files:
                raise ValueError("NPZ missing the 'notes' array")
            notes = data["notes"]
        if notes.ndim != 1 or notes.dtype.names is None:
            raise ValueError("'notes' must be a 1D structured NumPy array")
        missing = set(REQUIRED_FIELDS) - set(notes.dtype.names)
        if missing:
            raise ValueError(f"NPZ missing fields: {sorted(missing)}")
        return notes
    except Exception as exc:
        check.add("FAIL", "NPZ_READ", f"Cannot read {path.name}: {type(exc).__name__}: {exc}")
        return None


def compare_csv_npz(rows, array, check):
    if len(rows) != len(array):
        check.add("FAIL", "CSV_NPZ_COUNT", f"CSV has {len(rows)} rows, NPZ has {len(array)}")
    count = min(len(rows), len(array))
    mismatches = 0
    for i in range(count):
        for f in REQUIRED_FIELDS:
            a = rows[i][f]
            b = array[f][i]
            try:
                equal = (isclose(a, b) if f in FLOAT_FIELDS else
                         bool(a) == bool(b) if f in BOOL_FIELDS else
                         int(a) == int(b) if f in INT_FIELDS else
                         str(a) == str(b))
            except (TypeError, ValueError):
                equal = False
            if not equal:
                mismatches += 1
                check.capped("FAIL", "CSV_NPZ_MISMATCH", f"Row {i+1}, {f}: CSV {a!r} != NPZ {b!r}", record=i+1)
    return mismatches


def validate_numeric(rows, check):
    for i, r in enumerate(rows, 1):
        try:
            if not 0 <= r["pitch_midi"] <= 127:
                check.capped("FAIL", "PITCH", f"Invalid pitch {r['pitch_midi']}", record=i)
            if r["part_index"] < 1:
                check.capped("FAIL", "PART", "Part ID must be positive", record=i)
            if r["tied_segments"] < 1:
                check.capped("FAIL", "TIE_COUNT", "Tied-segment count must be >= 1", record=i)
            start, end, duration = (r[k] for k in ("onset_quarter", "offset_quarter", "duration_quarter"))
            if not all(math.isfinite(v) for v in (start, end, duration, r["measure_onset_quarter"])):
                check.capped("FAIL", "NONFINITE_TIME", "Nonfinite timing value", record=i)
                continue
            if end < start - TIME_TOL or duration < -TIME_TOL:
                check.capped("FAIL", "NEGATIVE_DURATION", f"onset={start}, offset={end}, duration={duration}", record=i)
            if not isclose(duration, end - start):
                check.capped("FAIL", "DURATION_ARITHMETIC", "Duration disagrees with offset minus onset", record=i)
            if r["offset_div"] < r["onset_div"]:
                check.capped("FAIL", "DIVISION_TIME", "Offset division precedes onset division", record=i)
            if not r["is_grace"] and (end <= start + TIME_TOL or r["offset_div"] == r["onset_div"]):
                check.capped("FAIL", "ORDINARY_ZERO_DURATION", "Ordinary note has zero duration", record=i)
            if r["is_grace"] and abs(duration) > TIME_TOL:
                check.capped("REVIEW", "GRACE_NONZERO", "Grace note has nonzero duration; inspect interpretation", record=i)
            if not r["measure_number"]:
                check.capped("FAIL", "NO_MEASURE", "Missing measure assignment", record=i)
            if r["measure_onset_quarter"] > start + TIME_TOL:
                check.capped("FAIL", "MEASURE_POSITION", "Measure starts after note onset", record=i)
            if r["source_file"] != check.source or r["timing_variant"] != check.variant:
                check.capped("FAIL", "ROW_PROVENANCE", "Source or variant disagrees with extraction summary", record=i)
            for frac_field, quarter_field in (("onset_whole_fraction", "onset_quarter"),
                                               ("offset_whole_fraction", "offset_quarter"),
                                               ("duration_whole_fraction", "duration_quarter")):
                try:
                    frac_value = float(Fraction(r[frac_field]))
                    if not isclose(frac_value, r[quarter_field] / 4, tol=2e-6):
                        check.capped("FAIL", "FRACTION", f"{frac_field}={r[frac_field]!r} disagrees with {quarter_field}", record=i)
                except (ValueError, ZeroDivisionError, OverflowError):
                    check.capped("FAIL", "FRACTION", f"Invalid {frac_field}={r[frac_field]!r}", record=i)
        except (TypeError, ValueError) as exc:
            check.capped("FAIL", "ROW_VALIDATION", f"Row {i}: {exc}", record=i)

    for i in range(1, len(rows)):
        if rows[i]["onset_quarter"] < rows[i-1]["onset_quarter"] - TIME_TOL:
            check.capped("FAIL", "UNSORTED", "Rows are not sorted by onset", record=i+1)
    return Counter({
        "notes": sum(not r["is_grace"] for r in rows),
        "grace_notes": sum(r["is_grace"] for r in rows),
        "tied_groups": sum(r["tied_segments"] > 1 for r in rows),
        "missing_measure": sum(not r["measure_number"] for r in rows),
        "missing_staff": sum(not r["staff"] for r in rows),
    })


def score_parts(score):
    if hasattr(score, "parts"):
        return list(score.parts)
    if hasattr(score, "notes_tied"):
        return [score]
    if isinstance(score, (tuple, list)):
        return [p for item in score for p in score_parts(item)]
    raise TypeError(f"Unsupported Partitura score object: {type(score)}")


def event_key(part_index, pitch, onset_div, offset_div, grace, ties, staff, voice):
    """Stable source/export identity (note IDs can be parser-generated)."""
    return (int(part_index), int(pitch), int(onset_div), int(offset_div),
            bool(grace), int(ties), str(staff), str(voice))


def count_tied_segments(note):
    count = 0
    seen = set()
    while note is not None:
        if id(note) in seen:
            raise ValueError("Cyclic tie chain in reloaded score")
        seen.add(id(note))
        count += 1
        note = getattr(note, "tie_next", None)
    return count


def compare_to_partitura(source_path, rows, selected, check, pt):
    """Independent source comparison; deliberately does NOT call the extractor."""
    try:
        failed_before = check.counts["FAIL"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if source_path.suffix.lower() in {".mid", ".midi"}:
                score = pt.load_score_midi(str(source_path), quantization_unit=1)
            else:
                score = pt.load_score(str(source_path))
        parts = score_parts(score)
        if max(selected) > len(parts):
            check.add("FAIL", "SOURCE_PARTS", f"Selected P{max(selected)} but reloaded score has {len(parts)} parts")
            return "FAILED"
        source_events = Counter()
        source_quarters = {}
        for part_id in selected:
            part = parts[part_id - 1]
            qmap = part.quarter_map
            for note in part.notes_tied:
                if note.start is None or note.end is None:
                    check.capped("FAIL", "SOURCE_NOTE_TIME", f"Part P{part_id} has note without an endpoint")
                    continue
                start = int(note.start.t)
                endpoint = getattr(note, "end_tied", None)
                end = int(endpoint.t if endpoint is not None else note.end.t)
                key = event_key(
                    part_id, note.midi_pitch, start, end,
                    isinstance(note, pt.score.GraceNote), count_tied_segments(note),
                    "" if getattr(note, "staff", None) is None else note.staff,
                    "" if getattr(note, "voice", None) is None else note.voice,
                )
                source_events[key] += 1
                source_quarters[key] = (float(qmap(start)), float(qmap(end)))
        export_events = Counter(event_key(
            r["part_index"], r["pitch_midi"], r["onset_div"], r["offset_div"],
            r["is_grace"], r["tied_segments"], r["staff"], r["voice"]
        ) for r in rows)
        missing = source_events - export_events
        extra = export_events - source_events
        if missing or extra:
            check.add("FAIL", "SOURCE_EVENT_MISMATCH",
                      f"Source has {sum(source_events.values())} selected note groups; "
                      f"export has {sum(export_events.values())}. "
                      f"Missing from export: {sum(missing.values())}; extra: {sum(extra.values())}. "
                      f"Examples missing={list(missing.items())[:3]}, extra={list(extra.items())[:3]}")
        for i, r in enumerate(rows, 1):
            key = event_key(r["part_index"], r["pitch_midi"], r["onset_div"],
                            r["offset_div"], r["is_grace"], r["tied_segments"],
                            r["staff"], r["voice"])
            if key in source_quarters:
                sq, eq = source_quarters[key]
                if not isclose(sq, r["onset_quarter"]) or not isclose(eq, r["offset_quarter"]):
                    check.capped("FAIL", "SOURCE_QUARTER_MISMATCH", "Quarter positions disagree with reloaded Partitura score", record=i)
        return ("MATCH" if not missing and not extra and
                check.counts["FAIL"] == failed_before else "MISMATCH")
    except Exception as exc:
        check.add("FAIL", "SOURCE_READ", f"Cannot independently compare with Partitura: {type(exc).__name__}: {exc}")
        return "FAILED"


def midi_raw_notes(source_path):
    """Match note_on/note_off per (track, channel, pitch) without changing MIDI."""
    import mido
    midi = mido.MidiFile(str(source_path))
    raw_tracks = []
    for idx, track in enumerate(midi.tracks):
        tick = 0
        active = defaultdict(deque)
        notes = []
        unmatched_offs = 0
        for message in track:
            tick += message.time
            if message.type not in ("note_on", "note_off"):
                continue
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                active[key].append(tick)
            elif active[key]:
                start = active[key].popleft()
                notes.append((message.note, start, tick))
            else:
                unmatched_offs += 1
        unmatched_ons = sum(len(q) for q in active.values())
        if notes or unmatched_offs or unmatched_ons:
            raw_tracks.append({"midi_track": idx, "notes": notes,
                               "unmatched_offs": unmatched_offs,
                               "unmatched_ons": unmatched_ons})
    return midi.ticks_per_beat, raw_tracks


def compare_raw_midi(source_path, rows, selected, check, imported_part_count):
    """Report raw-vs-imported MIDI differences; never alter or reject MIDI timing."""
    try:
        tpq, tracks = midi_raw_notes(source_path)
        if len(tracks) != imported_part_count:
            check.add("REVIEW", "MIDI_TRACK_MAPPING",
                      f"Found {len(tracks)} sounding raw MIDI tracks but {imported_part_count} Partitura parts; "
                      "cannot safely infer a 1-to-1 mapping")
            return "MAPPING_UNCLEAR"
        raw_events = Counter()
        unmatched_ons, unmatched_offs = 0, 0
        for part_id in selected:
            track = tracks[part_id-1]
            raw_events.update(track["notes"])
            unmatched_ons += track["unmatched_ons"]
            unmatched_offs += track["unmatched_offs"]
        exported = Counter((r["pitch_midi"], round(r["onset_quarter"] * tpq),
                            round(r["offset_quarter"] * tpq))
                           for r in rows if not r["is_grace"])
        missing, extra = raw_events - exported, exported - raw_events
        if missing or extra:
            check.add("REVIEW", "MIDI_EVENT_DIFFERENCE",
                      f"Original selected raw tracks have {sum(raw_events.values())} paired notes; "
                      f"export has {sum(exported.values())}. "
                      f"Raw-only={sum(missing.values())}, exported-only={sum(extra.values())}; "
                      f"examples raw-only={list(missing.items())[:3]}, "
                      f"exported-only={list(extra.items())[:3]}. "
                      "May reflect Partitura import/voice reconstruction; not corrected.")
        if unmatched_ons or unmatched_offs:
            check.add("REVIEW", "MIDI_UNMATCHED_EVENTS",
                      f"Selected raw MIDI tracks have unmatched note-on={unmatched_ons}, note-off={unmatched_offs}")
        return "MATCH" if not (missing or extra or unmatched_ons or unmatched_offs) else "DIFFERENCES"
    except ImportError:
        check.add("REVIEW", "MIDO_MISSING", "Install mido to compare original MIDI note events")
        return "SKIPPED"
    except Exception as exc:
        check.add("REVIEW", "MIDI_READ", f"Cannot compare raw MIDI: {type(exc).__name__}: {exc}")
        return "FAILED"


def locate_export(path_text, notes_dir):
    """Support moving the project since absolute paths were stored at extraction."""
    supplied = Path(path_text)
    if supplied.is_file():
        return supplied
    return notes_dir / supplied.name


def validate_one(item, notes_dir, score_dir, config, pt, skip_source):
    source, variant = item.get("source_file", ""), item.get("variant", "")
    check = Checks(source, variant)
    output = {"source_file": source, "variant": variant, "status": "FAIL",
              "selected_parts": item.get("selected_parts", ""),
              "ordinary_notes": "", "grace_notes": "", "tied_groups": "",
              "csv_npz_match": "NOT_CHECKED", "source_comparison": "NOT_CHECKED",
              "raw_midi_comparison": "NOT_APPLICABLE", "fail_count": 0,
              "review_count": 0, "info_count": 0}
    if item.get("status") != "OK":
        check.add("FAIL", "EXTRACTION_STATUS", f"Extraction summary status: {item.get('status')!r}; {item.get('error', '')}")
    if not source or Path(source).name != source:
        check.add("FAIL", "SOURCE_FILENAME", "Source filename is absent or contains a path")
    source_path = score_dir / source
    if not source_path.is_file():
        check.add("FAIL", "SOURCE_FILE_MISSING", f"Source score not found: {source_path}")
    if variant != "original":
        check.add("REVIEW", "TIMING_VARIANT", "Non-original timing variant selected for validation")
    try:
        selected = [int(s) for s in item.get("selected_parts", "").split(";") if s.strip()]
        if not selected or min(selected) < 1 or len(set(selected)) != len(selected):
            raise ValueError("Invalid selected parts")
    except ValueError:
        selected = []
        check.add("FAIL", "SELECTED_PARTS", "Invalid selected_parts in summary")

    selection = config.get(source)
    if selection is not None:
        if selection.get("parts") != selected:
            check.add("FAIL", "SELECTION_CONFIG", f"Summary selected {selected}; config selects {selection.get('parts')}")
        status = str(selection.get("status", ""))
        if "provisional" in status or "review" in status:
            check.add("REVIEW", "PART_SELECTION_PROVISIONAL", f"Part selection remains provisional: {selection.get('reason', '')}")
    elif selected != [1]:
        check.add("FAIL", "SELECTION_CONFIG", "Multiple-part selection missing from configuration")

    csv_text, npz_text = item.get("csv", ""), item.get("npz", "")
    if not csv_text or not npz_text:
        check.add("FAIL", "EXPORT_PATH", "CSV or NPZ path missing from summary")
        rows, array = None, None
    else:
        csv_path = locate_export(csv_text, notes_dir)
        npz_path = locate_export(npz_text, notes_dir)
        if not csv_path.is_file():
            check.add("FAIL", "EXPORT_NOT_FOUND", f"CSV not found: {csv_path}")
        if not npz_path.is_file():
            check.add("FAIL", "EXPORT_NOT_FOUND", f"NPZ not found: {npz_path}")
        rows = typed_csv_rows(csv_path, check) if csv_path.is_file() else None
        array = read_npz(npz_path, check) if npz_path.is_file() else None
    if rows is not None:
        stats = validate_numeric(rows, check)
        output["ordinary_notes"] = stats["notes"]
        output["grace_notes"] = stats["grace_notes"]
        output["tied_groups"] = stats["tied_groups"]
        actual_parts = sorted(set(r["part_index"] for r in rows))
        if not set(actual_parts).issubset(set(selected)):
            check.add("FAIL", "UNSELECTED_PART", f"Unexpected part IDs in export: {actual_parts}, expected subset {selected}")
        for label in ("notes", "grace_notes", "tied_groups", "missing_measure", "missing_staff"):
            try:
                if int(item[label]) != stats[label]:
                    check.add("FAIL", "SUMMARY_COUNT", f"{label}: summary={item[label]} export={stats[label]}")
            except (KeyError, ValueError):
                check.add("FAIL", "SUMMARY_COUNT", f"Cannot read summary count {label}")
        if stats["notes"] == 0:
            check.add("FAIL", "NO_ORDINARY_NOTES", "No ordinary notes in selected parts")
        if stats["missing_staff"]:
            if source_path.suffix.lower() in {".mid", ".midi"}:
                check.add("INFO", "MIDI_STAFF_UNAVAILABLE",
                          f"{stats['missing_staff']} MIDI records have no staff: expected MIDI metadata limitation")
            else:
                check.add("REVIEW", "MUSICXML_STAFF_UNAVAILABLE",
                          f"{stats['missing_staff']} MusicXML records lack staff labels")
        if rows and stats["grace_notes"]:
            check.add("INFO", "GRACE_PRESERVED", f"{stats['grace_notes']} grace notes preserved separately")
    if rows is not None and array is not None:
        mismatches = compare_csv_npz(rows, array, check)
        output["csv_npz_match"] = "YES" if mismatches == 0 and len(rows) == len(array) else "NO"

    if not skip_source and pt is not None and rows is not None and source_path.is_file() and selected and variant == "original":
        output["source_comparison"] = compare_to_partitura(source_path, rows, selected, check, pt)
    elif skip_source or pt is None:
        check.add("REVIEW", "SOURCE_COMPARISON_SKIPPED", "Partitura source comparison skipped; full validation incomplete")
        output["source_comparison"] = "SKIPPED"

    if (rows is not None and source_path.is_file() and selected and
            source_path.suffix.lower() in {".mid", ".midi"} and variant == "original"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = pt.load_score_midi(str(source_path), quantization_unit=1) if pt else None
            part_count = len(score_parts(score)) if score is not None else len(selected)
            output["raw_midi_comparison"] = compare_raw_midi(source_path, rows, selected, check, part_count)
        except Exception as exc:
            check.add("REVIEW", "MIDI_PARTS", f"MIDI part mapping unavailable: {type(exc).__name__}: {exc}")
            output["raw_midi_comparison"] = "SKIPPED"

    output["status"] = check.status
    output["fail_count"] = check.counts["FAIL"]
    output["review_count"] = check.counts["REVIEW"]
    output["info_count"] = check.counts["INFO"]
    return output, check.issues


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notes-dir", type=Path, default=NOTE_DIR,
                        help="Export folder with extraction_summary.csv, CSV and NPZ files")
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR,
                        help="Original score folder (default: PW/score)")
    parser.add_argument("--config", type=Path, default=CONFIG,
                        help="Score part selection JSON")
    parser.add_argument("--report-dir", type=Path, default=SCRIPT_DIR,
                        help="Write validation reports here (default: beside this script)")
    parser.add_argument("--skip-source-check", action="store_true",
                        help="Run export integrity checks only; source comparison marked incomplete")
    parser.add_argument("--include-variants", action="store_true",
                        help="Also inspect non-original experimental variants")
    args = parser.parse_args(argv)
    notes_dir, score_dir, report_dir = args.notes_dir.resolve(), args.score_dir.resolve(), args.report_dir.resolve()
    if not notes_dir.is_dir() or not (notes_dir / "extraction_summary.csv").is_file():
        parser.error(f"Missing extraction_summary.csv under {notes_dir}")
    if not score_dir.is_dir():
        parser.error(f"Original score directory not found: {score_dir}")
    try:
        with args.config.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
    except Exception as exc:
        parser.error(f"Cannot read part-selection configuration: {exc}")
    with (notes_dir / "extraction_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        entries = list(csv.DictReader(handle))
    if not args.include_variants:
        entries = [e for e in entries if e.get("variant") == "original"]
    if not entries:
        parser.error("Extraction summary contains no records to validate")

    pt = None
    if not args.skip_source_check:
        try:
            import partitura as pt
        except ImportError:
            parser.error("Partitura is required for full validation. Install it, or use --skip-source-check")

    report_dir.mkdir(parents=True, exist_ok=True)
    per_file = []
    issues = []
    originals = [e for e in entries if e.get("variant") == "original"]
    all_sources = {p.name for p in score_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED}
    reported_sources = [e.get("source_file", "") for e in originals]
    for missing in sorted(all_sources - set(reported_sources)):
        issues.append({"source_file": missing, "variant": "original", "severity": "FAIL",
                       "code": "MISSING_EXPORT", "record": "", "description": "No original extraction in summary"})
    for extra in sorted(set(reported_sources) - all_sources):
        issues.append({"source_file": extra, "variant": "original", "severity": "FAIL",
                       "code": "UNKNOWN_SOURCE", "record": "", "description": "Summary refers to a score absent from score/"})
    for name, count in Counter(reported_sources).items():
        if count > 1:
            issues.append({"source_file": name, "variant": "original", "severity": "FAIL",
                           "code": "DUPLICATE_SUMMARY", "record": "", "description": f"{count} original summary rows for one score"})

    print(f"Validating {len(entries)} exports, {len(all_sources)} original score files")
    print(f"Note exports: {notes_dir}")
    for e in entries:
        outcome, diagnostics = validate_one(e, notes_dir, score_dir, config, pt, args.skip_source_check)
        per_file.append(outcome)
        issues.extend(diagnostics)
        print(f"  [{outcome['status']:6}] {outcome['source_file']} "
              f"({outcome['ordinary_notes']} notes, {outcome['grace_notes']} grace; "
              f"source={outcome['source_comparison']}, raw MIDI={outcome['raw_midi_comparison']})")

    file_summary = report_dir / "note_validation_summary.csv"
    issue_file = report_dir / "note_validation_issues.csv"
    report_text = report_dir / "note_validation_report.txt"
    write_csv(file_summary, per_file, list(per_file[0]))
    write_csv(issue_file, issues,
              ["source_file", "variant", "severity", "code", "record", "description"])
    totals = Counter(x["status"] for x in per_file)
    issue_totals = Counter(x["severity"] for x in issues)
    full_failure = bool(issue_totals["FAIL"] or totals["FAIL"])
    with report_text.open("w", encoding="utf-8") as handle:
        handle.write("STEP 2.2d — NOTE-LEVEL EXTRACTION VALIDATION\n")
        handle.write(f"Scores in directory: {len(all_sources)}; exports checked: {len(entries)}\n")
        handle.write(f"Technical statuses: {dict(totals)}; issues: {dict(issue_totals)}\n")
        handle.write("PASS checks technical consistency, not musical correctness or audio alignment.\n")
        handle.write("REVIEW means an unresolved choice, raw-MIDI discrepancy, or skipped check.\n")
        handle.write("Original score and extracted note files were not modified.\n\n")
        for outcome in per_file:
            handle.write(f"[{outcome['status']}] {outcome['source_file']} ({outcome['variant']})\n")
            handle.write(f"    notes={outcome['ordinary_notes']}, grace={outcome['grace_notes']}, "
                         f"tied groups={outcome['tied_groups']}; "
                         f"CSV/NPZ={outcome['csv_npz_match']}, Partitura={outcome['source_comparison']}, "
                         f"raw MIDI={outcome['raw_midi_comparison']}\n")
            for issue in issues:
                if issue["source_file"] == outcome["source_file"] and issue["variant"] == outcome["variant"]:
                    handle.write(f"    {issue['severity']} {issue['code']}: {issue['description']}\n")
        coverage = [i for i in issues if i["code"] in ("MISSING_EXPORT", "UNKNOWN_SOURCE", "DUPLICATE_SUMMARY")]
        if coverage:
            handle.write("\nDATASET COVERAGE ISSUES:\n")
            for issue in coverage:
                handle.write(f"    {issue['severity']} {issue['source_file']}: {issue['description']}\n")
        handle.write("\nKnown external checks still deferred: comparison with printed notation, "
                     "repeat/performance order, excerpt alignment, and model performance.\n")

    print(f"\nStatuses: PASS={totals['PASS']}, REVIEW={totals['REVIEW']}, FAIL={totals['FAIL']}")
    print(f"Reports: {file_summary}, {issue_file}, {report_text}")
    if full_failure:
        print("Validation found integrity problems; inspect FAIL entries before Step 2.3.")
    else:
        print("No technical integrity failures. REVIEW findings are documented, not automatically corrected.")
    return 1 if full_failure else 0


if __name__ == "__main__":
    sys.exit(main())
