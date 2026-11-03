#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

EXTS = {".mxl", ".xml", ".musicxml", ".mid", ".midi"}
MIDI_EXTS = {".mid", ".midi"}
MEASURE_RE = re.compile(r"\bmeasure\s+(\d+)\b", re.IGNORECASE)
GRACE_TICK_RE = re.compile(r"\b(\d+)--\d+\s+GraceNote", re.IGNORECASE)

ISSUE_FIELDS = ["file", "priority", "category", "part", "measure", "evidence", "suggested_check"]
PART_FIELDS = ["file", "part_id", "part_name", "notes_written", "notes_tied_groups",
               "grace_notes", "measures", "first_measure", "last_measure",
               "pitch_min", "pitch_max", "staff_counts", "voice_counts",
               "first_onset_quarters", "last_offset_quarters"]
MEASURE_FIELDS = ["file", "part_id", "measure", "start_div", "end_div",
                  "note_onsets_in_measure", "grace_notes", "staff_counts", "voice_counts",
                  "raw_xml_notes", "raw_xml_graces", "raw_xml_backup_durations",
                  "raw_xml_forward_durations", "raw_xml_time_signatures", "notes_preview",
                  "grace_note_details"]
MIDI_FIELDS = ["file", "track", "channel", "pitch", "start_tick", "end_tick",
               "start_quarters", "end_quarters", "duration_quarters",
               "start_16th_error_ticks", "end_16th_error_ticks",
               "start_48th_error_ticks", "end_48th_error_ticks"]


def default_score_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "score", here / "score"):
        if candidate.is_dir():
            return candidate
    return here.parent / "score"


def txt(value: Any) -> str:
    return str(value) if value is not None else "—"


def brief(message: Any, limit: int = 450) -> str:
    return " ".join(str(message).split())[:limit]


def tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(node: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in node if tag_name(c.tag) == name]


def child_text(node: ET.Element, name: str) -> str:
    for c in node:
        if tag_name(c.tag) == name:
            return (c.text or "").strip()
    return ""


def safe_tick(obj: Any, boundary: str = "start") -> int | None:
    point = getattr(obj, boundary, None)
    t = getattr(point, "t", None)
    try:
        return int(t) if t is not None else None
    except (TypeError, ValueError):
        return None


def map_quarters(part: Any, tick: int | None) -> str:
    if tick is None:
        return ""
    try:
        return f"{float(part.quarter_map(tick)):.5f}"
    except (AttributeError, TypeError, ValueError):
        return ""


def measure_for_tick(part: Any, tick: int | None) -> str:
    if tick is None:
        return ""
    # Use explicit measures first, so missing meter does not crash diagnostics.
    measures = list(getattr(part, "measures", []))
    for i, m in enumerate(measures):
        start = safe_tick(m)
        end = safe_tick(m, "end")
        if start is not None and end is not None and start <= tick < end:
            return txt(getattr(m, "number", i + 1))
    if measures and tick == safe_tick(measures[-1], "end"):
        return txt(getattr(measures[-1], "number", len(measures)))
    return ""


def warning_category(message: str) -> str:
    m = message.lower()
    if "<backup> crosses measure boundary" in m:
        return "measure_boundary"
    if "grace note without" in m or "cadenza" in m:
        return "grace_or_cadenza"
    if "compositeduration" in m or "composite duration" in m:
        return "midi_duration_inference"
    if m.startswith("ignoring midi message"):
        return "midi_ignored_event"
    if (m.startswith("ignoring direction type:")
            or m.startswith('error parsing "')
            or m.startswith("not adding dynamics")
            or m.startswith("not adding duplicate or conflicting")
            or m.startswith("did not find a wedge")
            or m.startswith("did not find a pedal")
            or m.startswith("did not find a dashes")
            or m.startswith("dashes end")
            or m.startswith("dropping slur")):
        return "annotation"
    if m.strip() in {"pitch spelling", "create_part", "add notes", "add time sigs and measures",
                     "tie notes", "find tuplets", "done create_part", "add measures"}:
        return "midi_import_progress"
    return "unclassified"


def add_issue(out: list[dict], file: str, priority: str, category: str,
              evidence: str, suggestion: str, part: str = "", measure: str = "") -> None:
    out.append(dict(zip(ISSUE_FIELDS, [file, priority, category, part, measure,
                                      brief(evidence, 1200), suggestion])))


def load_score_xml(path: Path) -> ET.Element:
    """Read the MusicXML document, including the compressed MXL variant."""
    if path.suffix.lower() != ".mxl":
        return ET.parse(path).getroot()
    with zipfile.ZipFile(path) as zf:
        members = set(zf.namelist())
        xml_name = None
        container_name = next((n for n in members if n.lower() == "meta-inf/container.xml"), None)
        if container_name:
            container = ET.fromstring(zf.read(container_name))
            for node in container.iter():
                if tag_name(node.tag) == "rootfile":
                    xml_name = node.attrib.get("full-path")
                    if xml_name:
                        break
        if not xml_name:
            choices = sorted(n for n in members if n.lower().endswith(".xml")
                             and not n.lower().startswith("meta-inf/"))
            if len(choices) != 1:
                raise ValueError(f"Ambiguous MXL score XML entries: {choices[:8]}")
            xml_name = choices[0]
        if xml_name not in members:
            raise ValueError(f"MXL refers to missing XML entry {xml_name!r}")
        return ET.fromstring(zf.read(xml_name))


def raw_xml_summary(root: ET.Element) -> dict:
    """Unmodified XML facts indexed by (part ID, printed measure number)."""
    out = {}
    for part in root.iter():
        if tag_name(part.tag) != "part" or "id" not in part.attrib:
            continue
        part_id = part.attrib["id"]
        for i, measure in enumerate(children(part, "measure")):
            number = measure.attrib.get("number", str(i + 1))
            note_elems = children(measure, "note")
            raw_graces = sum(bool(children(n, "grace")) for n in note_elems)
            backups = [child_text(b, "duration") for b in children(measure, "backup")]
            forwards = [child_text(b, "duration") for b in children(measure, "forward")]
            sigs = []
            for attr in children(measure, "attributes"):
                for time in children(attr, "time"):
                    beats = child_text(time, "beats")
                    beat_type = child_text(time, "beat-type")
                    sigs.append(f"{beats}/{beat_type}")
            repeats = []
            for barline in children(measure, "barline"):
                for rep in children(barline, "repeat"):
                    repeats.append(rep.attrib.get("direction", "unknown"))
            out[(part_id, number)] = dict(notes=len(note_elems), graces=raw_graces,
                                          backups=backups, forwards=forwards,
                                          time_sigs=sigs, repeats=repeats)
    return out


def selected_notes(part: Any, measure: Any) -> list[Any]:
    start = safe_tick(measure)
    end = safe_tick(measure, "end")
    if start is None or end is None:
        return []
    return sorted([n for n in part.notes if safe_tick(n) is not None
                   and start <= safe_tick(n) < end],
                  key=lambda n: (safe_tick(n), getattr(n, "midi_pitch", 0)))


def note_short(note: Any) -> str:
    return (f"{txt(getattr(note, 'midi_pitch', '?'))}"
            f"@{txt(safe_tick(note))}..{txt(safe_tick(note, 'end'))}"
            f"(staff={txt(getattr(note, 'staff', None))},"
            f"voice={txt(getattr(note, 'voice', None))})")


def part_profile(file: str, part: Any) -> tuple[dict, str]:
    notes = list(part.notes)
    measures = list(part.measures)
    pitches = [int(n.midi_pitch) for n in notes if getattr(n, "midi_pitch", None) is not None]
    staff_counts = Counter(txt(getattr(n, "staff", None)) for n in notes)
    voice_counts = Counter(txt(getattr(n, "voice", None)) for n in notes)
    onsets = [safe_tick(n) for n in notes if safe_tick(n) is not None]
    ends = [safe_tick(n, "end") for n in notes if safe_tick(n, "end") is not None]
    fields = [file, txt(part.id), txt(part.part_name), len(notes),
              len(part.notes_tied), sum(type(n).__name__ == "GraceNote" for n in notes),
              len(measures), txt(getattr(measures[0], "number", None)) if measures else "",
              txt(getattr(measures[-1], "number", None)) if measures else "",
              min(pitches) if pitches else "", max(pitches) if pitches else "",
              str(dict(staff_counts)), str(dict(voice_counts)),
              map_quarters(part, min(onsets) if onsets else None),
              map_quarters(part, max(ends) if ends else None)]
    record = dict(zip(PART_FIELDS, fields))
    overview = (f"  {part.id}: {part.part_name or '(unnamed)'} | "
                f"notes={len(notes)}, tied groups={len(part.notes_tied)}, "
                f"measures={len(measures)}, pitch="
                f"{min(pitches) if pitches else '?'}..{max(pitches) if pitches else '?'} | "
                f"staff={dict(staff_counts)}, voice={dict(voice_counts)} | "
                f"quarter-time={record['first_onset_quarters']}..{record['last_offset_quarters']}")
    return record, overview


def measure_context(file: str, part: Any, measure: Any, raw: dict) -> tuple[dict, str]:
    num = txt(getattr(measure, "number", ""))
    notes = selected_notes(part, measure)
    staffs = Counter(txt(getattr(n, "staff", None)) for n in notes)
    voices = Counter(txt(getattr(n, "voice", None)) for n in notes)
    raw_info = raw.get((str(part.id), num), {})
    preview = "; ".join(note_short(n) for n in notes[:12])
    grace_details = "; ".join(note_short(n) for n in notes
                              if type(n).__name__ == "GraceNote")
    fields = [file, part.id, num, txt(safe_tick(measure)),
              txt(safe_tick(measure, "end")), len(notes),
              sum(type(n).__name__ == "GraceNote" for n in notes),
              str(dict(staffs)), str(dict(voices)),
              raw_info.get("notes", ""), raw_info.get("graces", ""),
              str(raw_info.get("backups", [])), str(raw_info.get("forwards", [])),
              ", ".join(raw_info.get("time_sigs", [])), preview, grace_details]
    record = dict(zip(MEASURE_FIELDS, fields))
    line = (f"    {part.id} measure {num}: "
            f"partitura [{safe_tick(measure)}, {safe_tick(measure, 'end')}), "
            f"{len(notes)} onset notes ({record['grace_notes']} grace), "
            f"raw XML notes={record['raw_xml_notes']}; backups="
            f"{record['raw_xml_backup_durations']}; "
            f"staves={dict(staffs)}; voices={dict(voices)}\n"
            f"      First notes: {preview or '(none)'}"
            + (f"\n      Grace notes: {grace_details}" if grace_details else ""))
    return record, line


def grid_error(tick: int, ppq: int, subdiv: int) -> float:
    step = ppq / subdiv
    return abs(tick / step - round(tick / step)) * step


def midi_diagnostics(path: Path, rel: str, issues: list[dict], max_examples: int):
    import mido
    midi = mido.MidiFile(str(path))
    ppq = int(midi.ticks_per_beat)
    if ppq <= 0:
        raise ValueError(f"MIDI has non-PPQ time division: {ppq}")
    raw_notes: list[tuple[int, int, int, int, int]] = []  # track, channel, pitch, start, end
    unmatched_off: list[tuple[int, int, int, int]] = []
    unmatched_on: list[tuple[int, int, int, int]] = []
    track_info = []
    for tr_id, track in enumerate(midi.tracks):
        tick = 0
        active: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        count = 0
        names = []
        channels = set()
        for msg in track:
            tick += int(msg.time)
            if msg.type == "track_name":
                names.append(str(msg.name))
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)].append(tick)
                channels.add(msg.channel)
                count += 1
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                channels.add(msg.channel)
                key = (msg.channel, msg.note)
                if active[key]:
                    start = active[key].popleft()
                    raw_notes.append((tr_id, msg.channel, msg.note, start, tick))
                else:
                    unmatched_off.append((tr_id, msg.channel, msg.note, tick))
        for (ch, pitch), onsets in active.items():
            unmatched_on.extend((tr_id, ch, pitch, start) for start in onsets)
        track_info.append((tr_id, " / ".join(names) or "(unnamed)", sorted(channels), count))

    lines = [f"  Raw MIDI: {len(midi.tracks)} tracks, {ppq} ticks/quarter, "
             f"{len(raw_notes)} matched notes; "
             f"unmatched on={len(unmatched_on)}, off={len(unmatched_off)}",
             "  Tracks:"]
    lines += [f"    {i}: name={name!r}, channels={channels}, note-ons={n}"
              for i, name, channels, n in track_info]
    if unmatched_off or unmatched_on:
        events = [f"OFF track={tr} ch={ch} pitch={p} tick={t}" for tr, ch, p, t in unmatched_off]
        events += [f"ON track={tr} ch={ch} pitch={p} tick={t}" for tr, ch, p, t in unmatched_on]
        lines.append("  Unmatched event examples: " + " | ".join(events[:max_examples]))
        add_issue(issues, rel, "CHECK", "midi_unmatched_events",
                  f"{len(unmatched_off)} unmatched offs; {len(unmatched_on)} unmatched ons. "
                  + " | ".join(events[:max_examples]),
                  "Inspect MIDI tracks/events; determine whether these are stray events or cross-track pairs.")

    samples: list[dict] = []
    if raw_notes:
        for subdiv, name in ((4, "16th"), (12, "48th")):
            for which, label in ((3, "onset"), (4, "offset")):
                distances = [grid_error(n[which], ppq, subdiv) for n in raw_notes]
                tolerance = max(1, ppq * .002)
                aligned = sum(d <= tolerance for d in distances)
                distances.sort()
                median = distances[len(distances) // 2]
                p90 = distances[min(len(distances) - 1, int(.9 * len(distances)))]
                lines.append(f"  {label} on {name} grid: {aligned}/{len(raw_notes)} "
                             f"({aligned / len(raw_notes):.1%}), median deviation="
                             f"{median:.2f} ticks, 90th percentile={p90:.2f} ticks")
        # Largest deviations from sixteenth-note grid; this is a *sample*, not a verdict.
        ranked = sorted(raw_notes, key=lambda n: grid_error(n[4], ppq, 4), reverse=True)
        for track, ch, pitch, start, end in ranked[:max_examples]:
            row = [rel, track, ch, pitch, start, end, f"{start / ppq:.5f}",
                   f"{end / ppq:.5f}", f"{(end - start) / ppq:.5f}",
                   f"{grid_error(start, ppq, 4):.2f}", f"{grid_error(end, ppq, 4):.2f}",
                   f"{grid_error(start, ppq, 12):.2f}", f"{grid_error(end, ppq, 12):.2f}"]
            samples.append(dict(zip(MIDI_FIELDS, row)))
        common = Counter(round((end - start) / ppq, 4)
                         for _, _, _, start, end in raw_notes if end > start)
        lines.append("  Most common raw durations (quarter notes, count): " +
                     str(common.most_common(10)))
        lines.append(f"  {len(samples)} largest-offset-deviation examples saved to midi_note_samples.csv")
        if len(raw_notes) and sum(grid_error(n[4], ppq, 4) <= max(1, ppq*.002)
                                  for n in raw_notes) / len(raw_notes) < .80:
            add_issue(issues, rel, "CHECK", "midi_articulated_offsets",
                      "Many raw MIDI note-offs are outside a 16th-note grid. "
                      f"Examples in midi_note_samples.csv; PPQ={ppq}.",
                      "Compare MIDI note releases with intended notation; do not round automatically.")
    return lines, samples


def analyze_file(path: Path, score_dir: Path, pt: Any, max_examples: int):
    rel = path.relative_to(score_dir).as_posix()
    is_midi = path.suffix.lower() in MIDI_EXTS
    issues: list[dict] = []
    parts_out: list[dict] = []
    measures_out: list[dict] = []
    midi_samples: list[dict] = []
    lines = ["=" * 90, f"FILE: {rel}", f"TYPE: {'MIDI' if is_midi else 'MusicXML'}"]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = pt.load_score(str(path))
        parts = list(score.parts)
    except Exception as exc:
        add_issue(issues, rel, "BLOCKER", "load_failure",
                  f"{type(exc).__name__}: {exc}", "Resolve parsing before note extraction.")
        lines.append(f"LOAD FAILED: {type(exc).__name__}: {exc}")
        return lines, issues, parts_out, measures_out, midi_samples

    parsed_warnings = [brief(w.message, 1200) for w in caught]
    categories = Counter(warning_category(m) for m in parsed_warnings)
    lines.append(f"Loaded {len(parts)} parts; parser warnings={len(caught)}: {dict(categories)}")
    lines.append("PART PROFILES:")
    for part in parts:
        rec, text = part_profile(rel, part)
        parts_out.append(rec)
        lines.append(text)

    if len(parts) > 1:
        # Multi-part is not an error; explicit selection is essential.
        add_issue(issues, rel, "CHECK", "part_selection",
                  f"{len(parts)} parts: " + "; ".join(f"{p.id}: {p.part_name}" for p in parts),
                  "Confirm piano part(s), both hands, and excerpt against audio; retain or reject parts explicitly.")
    for part in parts:
        if any(getattr(n, "staff", None) is None for n in part.notes):
            missing = sum(getattr(n, "staff", None) is None for n in part.notes)
            add_issue(issues, rel, "CHECK" if is_midi else "INFO", "staff_missing",
                      f"{missing}/{len(part.notes)} notes missing staff label in {part.id}.",
                      "Do not infer left/right hand from pitch alone.", part=str(part.id))
        if not list(part.time_sigs):
            add_issue(issues, rel, "CHECK", "no_time_signature",
                      f"Part {part.id} has no recovered time signatures.",
                      "Inspect source score, pickup and measure lengths; confirm meter explicitly.",
                      part=str(part.id))
        for repeat in part.repeats:
            onset = safe_tick(repeat)
            offset = safe_tick(repeat, "end")
            add_issue(issues, rel, "CHECK", "repeat",
                      f"Repeat {part.id}: {onset}..{offset} (score divisions), "
                      f"measures {measure_for_tick(part, onset)}..{measure_for_tick(part, offset)}",
                      "Compare repeat order against performance recording before unfolding.", part=str(part.id),
                      measure=measure_for_tick(part, onset))

    if is_midi:
        add_issue(issues, rel, "CHECK", "midi_score_import",
                  "Partitura inferred symbolic score/voices/measure structure from MIDI.",
                  "Verify note times, measures and voice/staff assignments against original notation where available.")
        try:
            midi_lines, midi_samples = midi_diagnostics(path, rel, issues, max_examples)
            lines.extend(midi_lines)
        except Exception as exc:
            lines.append(f"RAW MIDI INSPECTION FAILED: {type(exc).__name__}: {exc}")
            add_issue(issues, rel, "BLOCKER", "midi_raw_inspection_failed",
                      f"{type(exc).__name__}: {exc}", "Inspect MIDI file with mido or editor.")

    # Warning details, grouped by category; avoid burying the useful ones.
    if parsed_warnings:
        lines.append("PARTITURA WARNINGS (unique examples):")
        grouping: dict[str, Counter] = defaultdict(Counter)
        for msg in parsed_warnings:
            grouping[warning_category(msg)][msg] += 1
        for cat in ("measure_boundary", "grace_or_cadenza", "midi_duration_inference",
                    "midi_ignored_event", "unclassified", "annotation", "midi_import_progress"):
            if cat not in grouping:
                continue
            messages = grouping[cat]
            lines.append(f"  [{cat}] {sum(messages.values())} warnings, {len(messages)} distinct messages")
            cap = max_examples if cat not in ("annotation", "midi_import_progress") else 4
            for msg, count in messages.most_common(cap):
                lines.append(f"    ({count}×) {msg}")
            if len(messages) > cap:
                lines.append(f"    ... {len(messages) - cap} other distinct warnings omitted")
            if cat in ("measure_boundary", "grace_or_cadenza", "midi_duration_inference",
                       "midi_ignored_event", "unclassified"):
                category_action = {
                    "measure_boundary": "Inspect affected measure's XML backup/forward and parsed note onsets.",
                    "grace_or_cadenza": "Inspect grace-note attachment and local written ornament.",
                    "midi_duration_inference": "Check inferred symbolic durations; raw MIDI timing may still be usable.",
                    "midi_ignored_event": "Check event in raw MIDI tracks; compare with paired note data.",
                    "unclassified": "Classify this parser warning before treating it as harmless.",
                }[cat]
                # Each distinct meaningful warning retained in CSV up to 200 per file.
                for msg in list(messages)[:200]:
                    number = MEASURE_RE.search(msg)
                    m_no = number.group(1) if number else ""
                    add_issue(issues, rel, "CHECK" if cat != "unclassified" else "INVESTIGATE",
                              cat, msg, category_action, measure=m_no)
        if not (set(grouping) - {"annotation", "midi_import_progress"}):
            lines.append("  No note-timing/structure-specific Partitura warnings identified. "
                         "Annotation warnings alone do not imply broken pitches or durations.")

    raw: dict = {}
    if not is_midi:
        try:
            xml_root = load_score_xml(path)
            raw = raw_xml_summary(xml_root)
        except Exception as exc:
            lines.append(f"RAW MUSICXML INSPECTION FAILED: {type(exc).__name__}: {exc}")
            add_issue(issues, rel, "CHECK", "xml_read_failure",
                      f"{type(exc).__name__}: {exc}", "Examine compressed XML container and file integrity.")

    # In addition to numbered measure warnings, locate grace warnings using tick positions.
    target_measures: set[str] = set()
    for msg in parsed_warnings:
        if warning_category(msg) in ("measure_boundary", "grace_or_cadenza"):
            match = MEASURE_RE.search(msg)
            if match:
                target_measures.add(match.group(1))
            grace_tick = GRACE_TICK_RE.search(msg)
            if grace_tick:
                for part in parts:
                    mn = measure_for_tick(part, int(grace_tick.group(1)))
                    if mn:
                        target_measures.add(mn)
    # Always include measure 1 for missing explicit time signature; report how XML differs.
    if any(not list(p.time_sigs) for p in parts):
        target_measures.add("1")
    if target_measures:
        lines.append("AFFECTED MEASURE CONTEXT (Partitura and, where available, raw MusicXML):")
        matched = 0
        for part in parts:
            for measure in part.measures:
                number = txt(getattr(measure, "number", ""))
                if number not in target_measures:
                    continue
                row, text = measure_context(rel, part, measure, raw)
                measures_out.append(row)
                lines.append(text)
                matched += 1
        if not matched:
            lines.append(f"  No Partitura measures matched numeric warning labels: {sorted(target_measures)}")

    if not issues:
        lines.append("No action-specific issues detected. This is not proof of audio/score agreement.")
    else:
        levels = Counter(issue["priority"] for issue in issues)
        lines.append(f"ISSUE SUMMARY: {dict(levels)}; details in score_issues.csv")
    return lines, issues, parts_out, measures_out, midi_samples


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-dir", type=Path, default=default_score_dir(),
                        help="Score input folder (recursive; default: PW/score)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Destination folder for reports (default: beside this script)")
    parser.add_argument("--max-examples", type=int, default=20,
                        help="Maximum MIDI example notes and warnings per category (default: 20)")
    parser.add_argument("--file", action="append", default=[],
                        help="Inspect one named file (exact filename or relative path); repeatable")
    args = parser.parse_args(argv)
    score_dir = args.score_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not score_dir.is_dir():
        parser.error(f"Score directory not found: {score_dir}")
    if args.max_examples < 1:
        parser.error("--max-examples must be >= 1")
    try:
        import partitura as pt
    except ImportError:
        parser.error("Partitura not installed. Run: python -m pip install partitura")

    all_files = sorted((p for p in score_dir.rglob("*") if p.is_file()
                        and p.suffix.lower() in EXTS), key=lambda p: p.as_posix().casefold())
    if args.file:
        selection = set(args.file)
        files = [p for p in all_files if p.name in selection or
                 p.relative_to(score_dir).as_posix() in selection]
        not_found = selection - {p.name for p in files} - {
            p.relative_to(score_dir).as_posix() for p in files}
        if not_found:
            parser.error(f"Requested files not found: {sorted(not_found)}")
    else:
        files = all_files
    if not files:
        parser.error(f"No supported scores found in {score_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["TARGETED SCORE DIAGNOSTICS — Step 2.2a follow-up",
                    f"Scores: {score_dir}", f"Files: {len(files)}",
                    "Interpretation: CHECK/INVESTIGATE requests human validation, not an invalid file.",
                    "No changes to the original scores are made.", ""]
    all_issues = []
    all_parts = []
    all_measures = []
    all_midi_samples = []
    failures = 0
    for i, path in enumerate(files, start=1):
        rel = path.relative_to(score_dir).as_posix()
        print(f"[{i}/{len(files)}] {rel}", flush=True)
        try:
            lines, issues, parts, measures, samples = analyze_file(path, score_dir, pt, args.max_examples)
        except Exception as exc:
            failures += 1
            lines = ["=" * 90, f"FILE: {rel}",
                     f"DIAGNOSTIC ERROR: {type(exc).__name__}: {exc}"]
            issues, parts, measures, samples = [], [], [], []
            add_issue(issues, rel, "BLOCKER", "diagnostic_crash",
                      f"{type(exc).__name__}: {exc}", "Inspect traceback or report this script issue.")
        report_lines.extend(lines + [""])
        all_issues.extend(issues)
        all_parts.extend(parts)
        all_measures.extend(measures)
        all_midi_samples.extend(samples)

    destinations = {
        "score_diagnostics.txt": "\n".join(report_lines),
    }
    for fname, content in destinations.items():
        (output_dir / fname).write_text(content, encoding="utf-8")
    for filename, fields, rows in (
            ("score_issues.csv", ISSUE_FIELDS, all_issues),
            ("score_parts.csv", PART_FIELDS, all_parts),
            ("score_measure_details.csv", MEASURE_FIELDS, all_measures),
            ("midi_note_samples.csv", MIDI_FIELDS, all_midi_samples)):
        write_csv(output_dir / filename, fields, rows)

    counts = Counter(issue["priority"] for issue in all_issues)
    print(f"\nCompleted {len(files)} files; {len(all_parts)} parts, {len(all_measures)} targeted "
          f"measure rows, {len(all_issues)} issue rows; priorities: {dict(counts)}")
    print(f"Reports saved to: {output_dir}")
    print("  score_diagnostics.txt\n  score_issues.csv\n  score_parts.csv\n"
          "  score_measure_details.csv\n  midi_note_samples.csv")
    return 1 if failures or counts["BLOCKER"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
