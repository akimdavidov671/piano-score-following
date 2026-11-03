from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path
import sys
import warnings
import zipfile
from xml.etree import ElementTree as ET

try:
    import partitura as pt
except ImportError:
    sys.exit("Partitura is not installed. Run: python -m pip install partitura")

SUPPORTED_EXTENSIONS = {".mxl", ".musicxml", ".xml", ".mid", ".midi"}
MIDI_EXTENSIONS = {".mid", ".midi"}


def default_score_dir() -> Path:
    """Handle a script placed in either PW/ or PW/data-preparation/."""
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir.parent / "score", script_dir / "score"):
        if candidate.is_dir():
            return candidate
    return script_dir.parent / "score"


def percentage(part: int, total: int) -> str:
    return f"{100 * part / total:.1f}%" if total else "n/a"


def midi_timing_diagnostics(path: Path) -> dict:
    """Inspect original MIDI events, before score inference/quantization.

    A sixteenth-note grid is 1/4 quarter note; a 48th-note grid is
    1/12 quarter note (also accommodates common triplet subdivisions).
    These are heuristics, NOT proof of the source's rhythmic accuracy.
    """
    import mido  # Installed as a dependency of Partitura.

    midi = mido.MidiFile(str(path))
    ppq = midi.ticks_per_beat
    if ppq <= 0:
        raise ValueError(f"Invalid MIDI ticks_per_beat: {ppq}")

    raw_notes = []  # (onset_ticks, offset_ticks), within a given track
    unmatched_offs = 0
    unmatched_ons = 0
    for track in midi.tracks:
        absolute_tick = 0
        active = defaultdict(deque)  # (channel, pitch) -> queue of onsets
        for message in track:
            absolute_tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)].append(absolute_tick)
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                key = (message.channel, message.note)
                if active[key]:
                    start_tick = active[key].popleft()
                    raw_notes.append((start_tick, absolute_tick))
                else:
                    unmatched_offs += 1
        unmatched_ons += sum(len(queue) for queue in active.values())

    def on_grid(tick: int, subdivisions_per_quarter: int) -> bool:
        grid_ticks = ppq / subdivisions_per_quarter
        distance = abs(tick / grid_ticks - round(tick / grid_ticks)) * grid_ticks
        return distance <= max(1.0, ppq * 0.002)  # 1 tick / small rounding tolerance

    def fraction_on_grid(which: int, subdivisions: int) -> float | None:
        if not raw_notes:
            return None
        return sum(on_grid(note[which], subdivisions) for note in raw_notes) / len(raw_notes)

    durations = [end - start for start, end in raw_notes]
    common_durations = Counter(round(d / ppq, 4) for d in durations if d > 0)
    return {
        "midi_tracks": len(midi.tracks),
        "midi_ppq": ppq,
        "midi_raw_notes": len(raw_notes),
        "midi_unmatched_on": unmatched_ons,
        "midi_unmatched_off": unmatched_offs,
        "midi_bad_durations": sum(d <= 0 for d in durations),
        "midi_onset_16th": fraction_on_grid(0, 4),
        "midi_offset_16th": fraction_on_grid(1, 4),
        "midi_onset_48th": fraction_on_grid(0, 12),
        "midi_common_durations_quarters": common_durations.most_common(5),
    }


def classify_parser_warning(message: str) -> str:
    """Classify Partitura warnings for *this* note-onset/offset inspection.

    Classification is a diagnostic heuristic, not a guarantee that any
    particular MusicXML annotation has been interpreted correctly.
    """
    m = message.lower().strip()
    if (m.startswith("ignoring direction type:")
            or m.startswith("error parsing \"")
            or m.startswith("not adding duplicate or conflicting")
            or m.startswith("not adding dynamics indication")
            or m.startswith("did not find a wedge")
            or m.startswith("dropping slur")
            or m.startswith("did not find a dashes")):
        return "annotation"
    if m in {
        "pitch spelling", "create_part", "add notes",
        "add time sigs and measures", "tie notes", "find tuplets",
        "done create_part",
    }:
        return "midi_import_progress"
    if ("crosses measure boundary" in m or
            "grace note without" in m or
            "cadenza" in m or
            "compositeduration" in m or
            "duration" in m and "invalid" in m):
        return "timing_or_notation"
    if m.startswith("ignoring midi message"):
        return "midi_message"
    return "other"


def summarize_parser_warnings(caught: list) -> tuple[dict, list[str]]:
    """Return counts by category and short representative warning messages."""
    distinct = list(dict.fromkeys(str(w.message) for w in caught))
    # Count *emitted warnings*, not merely distinct wording.
    counts = Counter(classify_parser_warning(str(w.message)) for w in caught)
    # Display potentially consequential messages first.
    priority = {"timing_or_notation": 0, "midi_message": 1,
                "other": 2, "annotation": 3, "midi_import_progress": 4}
    distinct.sort(key=lambda m: priority[classify_parser_warning(m)])
    examples = [" ".join(m.split())[:150] for m in distinct[:5]]
    return dict(counts), examples



def validate_musicxml_independently(path: Path) -> tuple[str, str]:
    """Check XML well-formedness *without Partitura* (not MusicXML schema validity).

    For compressed MusicXML (.mxl), read the MusicXML document inside the
    ZIP archive according to META-INF/container.xml when available.
    Returns (WELL_FORMED | MALFORMED | UNDETERMINED, explanation).
    """
    try:
        if path.suffix.lower() == ".mxl":
            with zipfile.ZipFile(path) as archive:
                xml_names = [name for name in archive.namelist()
                             if name.lower().endswith(".xml")
                             and not name.lower().startswith("meta-inf/")]
                score_entry = None
                if "META-INF/container.xml" in archive.namelist():
                    metadata = ET.fromstring(archive.read("META-INF/container.xml"))
                    for rootfile in metadata.iter():
                        if rootfile.tag.rsplit("}", 1)[-1] == "rootfile":
                            score_entry = rootfile.attrib.get("full-path")
                            if score_entry:
                                break
                if not score_entry:
                    if len(xml_names) != 1:
                        return ("UNDETERMINED", f"Could not uniquely identify score XML: {xml_names[:5]}")
                    score_entry = xml_names[0]
                xml_bytes = archive.read(score_entry)
                entry_label = score_entry
        else:
            xml_bytes = path.read_bytes()
            entry_label = path.name

        try:
            ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            line, column = exc.position
            lines = xml_bytes.decode("utf-8", errors="replace").splitlines()
            excerpt = lines[line - 1].strip()[:160] if 0 < line <= len(lines) else ""
            return ("MALFORMED", f"{entry_label}: {exc}; near line {line}: {excerpt!r}")
        return ("WELL_FORMED", f"{entry_label}: independent XML syntax check passed")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
        return ("UNDETERMINED", f"Independent check could not complete: {type(exc).__name__}: {exc}")


def inspect_score(path: Path, root: Path) -> dict:
    relative_path = path.relative_to(root).as_posix()
    result = {
        "file": relative_path,
        "format": "MIDI" if path.suffix.lower() in MIDI_EXTENSIONS else "MusicXML",
        "status": "OK",
        "parts": 0,
        "part_names": "",
        "measures_by_part": "",
        "notes_written": 0,
        "notes_tied": 0,
        "grace_notes": 0,
        "pitch_range": "",
        "invalid_pitch": 0,
        "time_signatures": "",
        "staff_present": "",
        "voice_present": "",
        "nonpositive_duration": 0,
        "repeats": 0,
        "warnings": "",  # Actionable review reasons only.
        "parser_warning_count": 0,
        "parser_warning_types": "",
        "parser_warning_examples": "",
        "error": "",
        "xml_check": "",
        "xml_check_detail": "",
        "midi_tracks": "",
        "midi_ppq": "",
        "midi_raw_notes": "",
        "midi_unmatched_on": "",
        "midi_unmatched_off": "",
        "midi_bad_durations": "",
        "midi_onset_16th": "",
        "midi_offset_16th": "",
        "midi_onset_48th": "",
        "midi_common_durations_quarters": "",
    }
    is_midi = path.suffix.lower() in MIDI_EXTENSIONS
    messages = []  # Explicit review reasons, not every parser diagnostic.

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = pt.load_score(str(path))
        # Keep parser diagnostics separate from review reasons. Annotation
        # warnings do not, by themselves, invalidate pitch/onset extraction.
        warning_counts, warning_examples = summarize_parser_warnings(caught)
        result["parser_warning_count"] = len(caught)
        result["parser_warning_types"] = "; ".join(
            f"{name}={count}" for name, count in sorted(warning_counts.items())
        )
        result["parser_warning_examples"] = " | ".join(warning_examples)
        structural_count = sum(warning_counts.get(name, 0) for name in
                               ("timing_or_notation", "midi_message", "other"))
        if structural_count:
            messages.append(
                f"{structural_count} parser warning(s) potentially affecting "
                "note timing/notation; inspect examples"
            )
        parts = list(score.parts)
        result["parts"] = len(parts)
        result["part_names"] = "; ".join(
            f"{p.id}: {p.part_name or '(unnamed)'}" for p in parts
        )
        if len(parts) > 2:
            messages.append(
                f"{len(parts)} score parts: confirm which belong to the piano "
                "and which parts match the recording"
            )
        if not parts:
            messages.append("No score parts found")

        measure_counts = []
        signatures = set()
        staff_count = voice_count = 0
        all_notes = []
        valid_pitches = []
        for part in parts:
            notes = list(part.notes)
            all_notes.extend(notes)
            result["notes_written"] += len(notes)
            result["notes_tied"] += len(part.notes_tied)
            measure_counts.append(len(part.measures))
            result["grace_notes"] += sum(isinstance(n, pt.score.GraceNote) for n in notes)
            result["repeats"] += len(part.repeats)
            for ts in part.time_sigs:
                signatures.add(f"{ts.beats}/{ts.beat_type}")
            for note in notes:
                pitch = getattr(note, "midi_pitch", None)
                if pitch is None or not (0 <= pitch <= 127):
                    result["invalid_pitch"] += 1
                else:
                    valid_pitches.append(int(pitch))
                staff_count += getattr(note, "staff", None) is not None
                voice_count += getattr(note, "voice", None) is not None
                if not isinstance(note, pt.score.GraceNote):
                    if note.start is None or note.end is None or note.end.t <= note.start.t:
                        result["nonpositive_duration"] += 1

        result["pitch_range"] = (f"{min(valid_pitches)}–{max(valid_pitches)}"
                                 if valid_pitches else "not available")
        result["measures_by_part"] = ",".join(map(str, measure_counts))
        result["time_signatures"] = ", ".join(sorted(signatures)) or "not encoded"
        result["staff_present"] = percentage(staff_count, len(all_notes))
        result["voice_present"] = percentage(voice_count, len(all_notes))
        if all_notes and staff_count == 0:
            messages.append("Staff labels missing; do not assume staff assignments")
        if all_notes and voice_count == 0:
            messages.append("Voice labels missing; do not assume voice assignments")

        if not all_notes:
            messages.append("No pitched notes loaded")
        if result["invalid_pitch"]:
            messages.append(f"{result['invalid_pitch']} notes have missing or invalid MIDI pitches")
        if not any(measure_counts):
            messages.append("No explicit measures loaded")
        if not signatures:
            messages.append("No explicit time signatures loaded")
        if result["nonpositive_duration"]:
            messages.append(f"{result['nonpositive_duration']} non-grace notes have invalid durations")
        if result["repeats"]:
            messages.append("Repeat markings present: performance order needs checking")

        if is_midi:
            # Report underlying timing separately from inferred score structure.
            midi_stats = midi_timing_diagnostics(path)
            result.update(midi_stats)
            messages.append("MIDI: verify score-style timing and inferred notation manually")
            if midi_stats["midi_unmatched_on"] or midi_stats["midi_unmatched_off"]:
                messages.append("MIDI contains unmatched note-on/note-off events")
            # Written note segments can exceed MIDI notes when ties are
            # reconstructed. Compare against tied note groups instead.
            if (midi_stats["midi_raw_notes"] and
                    midi_stats["midi_raw_notes"] != result["notes_tied"]):
                messages.append(
                    "Raw MIDI note count differs from parsed tied-note groups; "
                    "inspect note handling"
                )
            if (midi_stats["midi_onset_16th"] is not None and
                    midi_stats["midi_onset_16th"] < 0.90 and
                    midi_stats["midi_onset_48th"] < 0.90):
                messages.append("MIDI onsets often fall outside tested rhythmic grids")

        result["warnings"] = " | ".join(dict.fromkeys(messages))
        if messages:
            result["status"] = "REVIEW"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["warnings"] = " | ".join(messages)
        if not is_midi:
            # A Partitura parsing error alone does NOT prove invalid source XML.
            # Independently verify XML syntax before advising replacement.
            status, detail = validate_musicxml_independently(path)
            result["xml_check"] = status
            result["xml_check_detail"] = detail
    return result


def print_result(row: dict) -> None:
    print(f"\n[{row['status']}] {row['file']} ({row['format']})")
    if row["error"]:
        print(f"  ERROR: {row['error']}")
        if row["xml_check"]:
            print(f"  Independent XML check: {row['xml_check']}")
            print(f"  Detail: {row['xml_check_detail']}")
        return
    print(f"  Parts: {row['parts']} | Measures per part: {row['measures_by_part'] or 'none'}")
    if row["parts"] > 1:
        print(f"  Part names: {row['part_names']}")
    print(f"  Notes (written/tied groups): {row['notes_written']} / {row['notes_tied']}"
          f" | Grace: {row['grace_notes']} | Repeats: {row['repeats']}")
    print(f"  MIDI pitch range: {row['pitch_range']}")
    print(f"  Time signatures: {row['time_signatures']}")
    print(f"  Note labels present — staff: {row['staff_present']}, voice: {row['voice_present']}")
    if row["nonpositive_duration"]:
        print(f"  Invalid non-grace durations: {row['nonpositive_duration']}")
    if row["format"] == "MIDI":
        print(f"  Raw MIDI: {row['midi_tracks']} tracks, {row['midi_ppq']} ticks/quarter, "
              f"{row['midi_raw_notes']} matched notes")
        print(f"  Onsets near 16th-note grid: {percentage_fraction(row['midi_onset_16th'])}"
              f" | 48th-note grid: {percentage_fraction(row['midi_onset_48th'])}")
        print(f"  Offsets near 16th-note grid: {percentage_fraction(row['midi_offset_16th'])}")
        print(f"  Common note durations (quarter notes, count): "
              f"{row['midi_common_durations_quarters']}")
        print(f"  Unmatched events — note-on: {row['midi_unmatched_on']}, "
              f"note-off: {row['midi_unmatched_off']}")
    if row["parser_warning_count"]:
        print(f"  Partitura diagnostics: {row['parser_warning_count']} "
              f"({row['parser_warning_types']})")
    if row["warnings"]:
        for message in row["warnings"].split(" | "):
            print(f"  Review: {message}")


def percentage_fraction(value) -> str:
    return "n/a" if value is None or value == "" else f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-dir", type=Path, default=default_score_dir(),
                        help="Folder containing source scores (searched recursively)")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV destination (default: beside inspect_scores.py)")
    parser.add_argument("--no-csv", action="store_true", help="Only print to terminal")
    args = parser.parse_args()
    root = args.score_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Score directory does not exist: {root}")
    files = sorted((p for p in root.rglob("*") if p.is_file() and
                    p.suffix.lower() in SUPPORTED_EXTENSIONS),
                   key=lambda p: str(p).lower())
    if not files:
        print(f"No supported score files found in {root}")
        return 1

    print(f"Inspecting {len(files)} scores in {root}")
    print("NOTE: REVIEW means manual checking is advisable, not that the score is invalid.\n      Cosmetic Partitura annotation warnings are counted, but do not trigger REVIEW.")
    rows = []
    for path in files:
        row = inspect_score(path, root)
        rows.append(row)
        print_result(row)

    totals = Counter(row["status"] for row in rows)
    print(f"\nSummary: {len(rows)} files | OK: {totals['OK']} | "
          f"REVIEW: {totals['REVIEW']} | ERROR: {totals['ERROR']}")
    if not args.no_csv:
        dest = (args.output or (Path(__file__).resolve().parent / "inspection_report.csv")).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV report: {dest}")
    return 1 if totals["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
