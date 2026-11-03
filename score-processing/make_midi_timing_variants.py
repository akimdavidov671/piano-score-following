from __future__ import annotations

import argparse
import csv
import json
import math
import re
from bisect import bisect_right
from fractions import Fraction
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import extract_score_notes as extract

SCRIPT_DIR = Path(__file__).resolve().parent
MOZART_FILENAME = "mozart_piano_concerto_17_453_1_(nc)fisher.mid"
SONATA_FILES = (
    "sonata_24_1_(c)iscenko.mid",
    "sonata_24_2_(c)iscenko.mid",
    "sonata_24_3_(c)iscenko.mid",
)
SOURCES = (MOZART_FILENAME, *SONATA_FILES)
SONATA_TPQ = 96
SONATA_GRID_TICKS = 24  # Sixteenth note (96/4); observed onsets mostly lie here
SONATA_MAX_EXTENSION_TICKS = 8  # 1/48 of a whole note, < 1/32 note
SONATA_VARIANT = "sonata_gap8_v1"
MOZART_VARIANT = "mozart_plus4"
DEFAULT_SCORE_DIR = SCRIPT_DIR.parent / "score"
DEFAULT_NOTES_DIR = SCRIPT_DIR / "score-notes"
DEFAULT_OUTPUT_DIR = DEFAULT_NOTES_DIR / "timing-adjusted"
DEFAULT_CONFIG = SCRIPT_DIR / "score_part_selection.json"


def adjust_mozart_midi(input_path: Path):
    """Create a *temporary in-memory* MIDI variant with selective +4-tick offs.

    Very deliberately conservative: modify only Mozart's raw tracks 1 and 2,
    only fully paired notes starting on a 48th-quarter grid, and only when the
    duration is exactly 4 ticks below a multiple of one 12th-quarter. Never
    extend a note beyond the next onset of that pitch/channel/track.
    No general quantization or re-estimation of scores is attempted.
    """
    import mido

    midi = mido.MidiFile(str(input_path))
    tpq = midi.ticks_per_beat
    if tpq != 384:
        raise ValueError(f"Mozart-specific correction expects 384 ticks/quarter, found {tpq}")
    stats = {"candidate": 0, "corrected": 0, "skipped_overlap": 0,
             "ambiguous_overlap": 0, "unmatched_events": 0, "ticks_added_per_note": 4,
             "tracks": [1, 2], "ticks_per_quarter": tpq}
    audit = []
    for track_idx in (1, 2):
        if track_idx >= len(midi.tracks):
            raise ValueError(f"Missing Mozart MIDI track {track_idx}")
        abs_tick = 0
        events = []
        active = defaultdict(list)
        paired = []
        for idx, msg in enumerate(midi.tracks[track_idx]):
            abs_tick += msg.time
            item = {"tick": abs_tick, "order": idx, "msg": msg}
            events.append(item)
            if msg.type not in ("note_on", "note_off"):
                continue
            key = (msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                active[key].append(item)
            elif active[key]:
                onset = active[key].pop(0)
                paired.append((onset, item, key))
            else:
                stats["unmatched_events"] += 1
        stats["unmatched_events"] += sum(len(a) for a in active.values())
        onsets_by_key = defaultdict(list)
        for on, off, key in paired:
            onsets_by_key[key].append(on["tick"])
        for key in onsets_by_key:
            onsets_by_key[key].sort()
        for on, off, key in paired:
            duration = off["tick"] - on["tick"]
            # 384/48 = 8 ticks, 384/12 = 32 ticks. The extra four ticks
            # must *exactly* recover a plausible metrical note duration.
            if (duration <= 0 or on["tick"] % 8 or
                    (duration + 4) % 32 != 0):
                continue
            stats["candidate"] += 1
            # Avoid creating a same-pitch overlap; use the first *later* onset.
            future = [t for t in onsets_by_key[key] if t > on["tick"]]
            next_onset = future[0] if future else None
            if next_onset is not None and off["tick"] + 4 > next_onset:
                stats["skipped_overlap"] += 1
                continue
            if onsets_by_key[key].count(on["tick"]) > 1:
                stats["ambiguous_overlap"] += 1
                continue
            audit.append({
                "raw_midi_track": track_idx, "channel": key[0], "pitch_midi": key[1],
                "onset_tick": on["tick"], "original_offset_tick": off["tick"],
                "corrected_offset_tick": off["tick"] + 4,
            })
            off["tick"] += 4
            stats["corrected"] += 1
        # Restore sorted nonnegative MIDI delta times, moving EOT to the end.
        def priority(item):
            msg = item["msg"]
            if msg.type == "end_of_track":
                return 3
            if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                return 0
            if msg.type == "note_on" and msg.velocity > 0:
                return 2
            return 1
        # Keep end-of-track at or after every possibly-extended note event.
        non_eot_end = max((e["tick"] for e in events
                           if e["msg"].type != "end_of_track"), default=0)
        for e in events:
            if e["msg"].type == "end_of_track":
                e["tick"] = max(e["tick"], non_eot_end)
        events.sort(key=lambda e: (e["tick"], priority(e), e["order"]))
        prior = 0
        track_out = mido.MidiTrack()
        for e in events:
            delta = e["tick"] - prior
            if delta < 0:
                raise RuntimeError("Negative MIDI delta while building correction variant")
            track_out.append(e["msg"].copy(time=delta))
            prior = e["tick"]
        midi.tracks[track_idx] = track_out
    return midi, stats, audit



def _is_note_on(msg):
    return msg.type == "note_on" and msg.velocity > 0


def _is_note_off(msg):
    return msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)


def _events_and_pairs(midi, track_ids):
    """Read raw MIDI in absolute ticks. Pair by track/channel/pitch FIFO.

    FIFO is a representation choice for overlapping same-pitch events. Such
    ambiguous events are rejected as adjustment candidates below.
    """
    all_events = {}
    pairs = []
    unmatched_on = unmatched_off = 0
    for track_id in track_ids:
        if track_id >= len(midi.tracks):
            raise ValueError(f"Raw MIDI track {track_id} does not exist")
        tick = 0
        track_events = []
        active = defaultdict(list)
        for order, message in enumerate(midi.tracks[track_id]):
            tick += message.time
            item = {"tick": tick, "order": order, "msg": message, "track": track_id}
            track_events.append(item)
            if _is_note_on(message):
                active[(message.channel, message.note)].append(item)
            elif _is_note_off(message):
                key = (message.channel, message.note)
                if active[key]:
                    onset = active[key].pop(0)
                    pairs.append((onset, item, key))
                else:
                    unmatched_off += 1
        unmatched_on += sum(len(queue) for queue in active.values())
        all_events[track_id] = track_events
    return all_events, pairs, unmatched_on, unmatched_off


def _restore_track(midi, track_idx, events):
    """Recalculate delta times without reordering unchanged simultaneous events.

    A moved note-off previously appeared before its target note-on in the raw
    track, so sorting by (absolute_tick, ORIGINAL order) puts it before that
    onset without disturbing the ordering of other same-tick messages.
    """
    import mido

    end = max((event["tick"] for event in events
               if event["msg"].type != "end_of_track"), default=0)
    for event in events:
        if event["msg"].type == "end_of_track":
            event["tick"] = max(event["tick"], end)
    events.sort(key=lambda item: (item["tick"],
                                  item["msg"].type == "end_of_track",
                                  item["order"]))
    rebuilt = mido.MidiTrack()
    previous = 0
    for event in events:
        delta = event["tick"] - previous
        if delta < 0:
            raise AssertionError("Unexpected negative MIDI delta after correction")
        rebuilt.append(event["msg"].copy(time=delta))
        previous = event["tick"]
    midi.tracks[track_idx] = rebuilt


def adjust_sonata_midi(input_path: Path):
    """Make one conservative timing variant using actual note-onset boundaries.

    Only raw tracks 1, 2, 3 (the selected P1, P2, P3) are considered. A note
    qualifies if its onset is on the 24-tick rhythmic grid and its offset is
    1..8 ticks before the earliest later *observed* selected-track note onset
    (also on that grid). We never choose an unobserved quantization target.
    We skip any intervening other note-off, same-pitch onset or ambiguity.
    """
    import mido

    midi = mido.MidiFile(str(input_path))
    if midi.ticks_per_beat != SONATA_TPQ:
        raise ValueError(f"Sonata correction expects {SONATA_TPQ} ticks/quarter; "
                         f"found {midi.ticks_per_beat} in {input_path.name}")
    track_ids = (1, 2, 3)
    track_events, pairs, unmatched_on, unmatched_off = _events_and_pairs(midi, track_ids)
    onsets = sorted({event["tick"] for events in track_events.values()
                     for event in events if _is_note_on(event["msg"])})
    # Protect all note-off boundaries (including unpaired events) in selected tracks.
    offs = sorted({event["tick"] for events in track_events.values()
                   for event in events if _is_note_off(event["msg"])})
    pitch_onsets = defaultdict(list)
    exact_onsets = Counter()
    for events in track_events.values():
        for event in events:
            if _is_note_on(event["msg"]):
                pitch_onsets[event["msg"].note].append(event["tick"])
                exact_onsets[(event["track"], event["msg"].channel,
                              event["msg"].note, event["tick"])] += 1
    for times in pitch_onsets.values():
        times.sort()

    stats = dict(ticks_per_quarter=SONATA_TPQ,
                 max_extension_ticks=SONATA_MAX_EXTENSION_TICKS,
                 source_pairs=len(pairs), unmatched_on=unmatched_on,
                 unmatched_off=unmatched_off, candidates=0, corrected=0,
                 skipped_intervening_off=0, skipped_same_pitch=0,
                 skipped_ambiguous=0, skipped_unaligned_onset=0,
                 skipped_unaligned_target=0)
    audit = []
    for start, stop, (channel, pitch) in pairs:
        onset = start["tick"]
        original_offset = stop["tick"]
        if original_offset <= onset:
            continue
        if onset % SONATA_GRID_TICKS:
            stats["skipped_unaligned_onset"] += 1
            continue
        j = bisect_right(onsets, original_offset)
        if j == len(onsets):
            continue
        target = onsets[j]
        gap = target - original_offset
        if gap < 1 or gap > SONATA_MAX_EXTENSION_TICKS:
            continue
        if target % SONATA_GRID_TICKS:
            stats["skipped_unaligned_target"] += 1
            continue
        stats["candidates"] += 1
        if exact_onsets[(start["track"], channel, pitch, onset)] > 1:
            stats["skipped_ambiguous"] += 1
            continue
        # Avoid overlapping a later same-pitch attack, even if that attack
        # already overlaps the candidate note in the original source.
        same_pitch_times = pitch_onsets[pitch]
        k = bisect_right(same_pitch_times, onset)
        if k < len(same_pitch_times) and same_pitch_times[k] < target:
            stats["skipped_same_pitch"] += 1
            continue
        # An onset before 'target' is impossible since target is the first one.
        # But crossing ANY intervening note-off may erase a meaningful boundary.
        off_j = bisect_right(offs, original_offset)
        if off_j < len(offs) and offs[off_j] < target:
            stats["skipped_intervening_off"] += 1
            continue
        audit.append({
            "raw_midi_track": start["track"], "channel": channel,
            "pitch_midi": pitch, "onset_tick": onset,
            "original_offset_tick": original_offset,
            "corrected_offset_tick": target,
            "extension_ticks": gap, "target_onset_tick": target,
            "correction_rule": "next_onset_grid24_gap_le8_no_intervening_events",
        })
        stop["tick"] = target
        stats["corrected"] += 1
    if stats["corrected"] != len(audit):
        raise AssertionError("Raw event/audit count mismatch")
    for track_id, events in track_events.items():
        _restore_track(midi, track_id, events)
    return midi, stats, audit


def read_baseline(path: Path, source_file: str):
    """Read previously validated original extracted CSV. Never modify it."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(extract.FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing baseline note columns in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty baseline note export: {path}")
    if any(row["source_file"] != source_file or row["timing_variant"] != "original"
           for row in rows):
        raise ValueError(f"Baseline source/variant mismatch: {path}")
    return rows


def compare_to_baseline(baseline: list[dict], adjusted: list[dict],
                        tpq: int, max_ticks: int):
    """Verify pitch, part, onset and note counts are unchanged.

    For duplicate notes with identical pitches/onsets, compare offset multisets
    in sorted order. All actual changes must be positive integer tick amounts
    up to the file-specific limit. Fails closed on reconstruction differences.
    """
    def key(row):
        return (int(row["part_index"]), int(row["pitch_midi"]),
                round(float(row["onset_quarter"]), 9),
                str(row["is_grace"]).lower() in ("true", "1"))

    orig = defaultdict(list)
    new = defaultdict(list)
    for row in baseline:
        orig[key(row)].append(float(row["offset_quarter"]))
    for row in adjusted:
        new[key(row)].append(float(row["offset_quarter"]))
    counts1 = Counter({k: len(v) for k, v in orig.items()})
    counts2 = Counter({k: len(v) for k, v in new.items()})
    if counts1 != counts2:
        removed = list((counts1 - counts2).items())[:5]
        added = list((counts2 - counts1).items())[:5]
        raise ValueError(f"Re-import altered counts, pitches, parts or onsets: "
                         f"missing={removed}; added={added}")
    changes = []
    eps = 1e-7
    for group_key in orig:
        for before, after in zip(sorted(orig[group_key]), sorted(new[group_key])):
            delta_ticks = (after - before) * tpq
            if abs(delta_ticks) <= eps:
                continue
            if (delta_ticks < -eps or delta_ticks > max_ticks + eps or
                    not math.isclose(delta_ticks, round(delta_ticks), abs_tol=eps)):
                raise ValueError(f"Unexpected offset difference at {group_key}: "
                                 f"{before} -> {after}, delta_ticks={delta_ticks}")
            changes.append((group_key, round(before * tpq),
                            round(after * tpq), int(round(delta_ticks))))
    return changes


def _state_diagnostics(rows: list[dict]):
    """Evaluate downstream state effects without saving/altering any states.

    Uses Step 2.3's unmodified state-construction algorithm. Results are
    diagnostics only; fewer states does not establish better alignment.
    """
    from build_score_states import build_states

    notes = [(int(row["pitch_midi"]),
              Fraction(str(row["onset_whole_fraction"])) * 4,
              Fraction(str(row["offset_whole_fraction"])) * 4)
             for row in rows if str(row["is_grace"]).lower() not in ("true", "1")]
    states = build_states(notes)
    short = sum(Fraction(s["length_whole_fraction"]) * 4 < Fraction(1, 32)
                for s in states)
    return {
        "state_count": len(states),
        "rest_count": sum(s["is_rest"] for s in states),
        "short_count": short,
    }


def _audit_columns():
    return ("source_file", "variant", "raw_midi_track", "channel", "pitch_midi",
            "onset_tick", "original_offset_tick", "corrected_offset_tick",
            "extension_ticks", "target_onset_tick", "correction_rule")


def _atomic_csv(path: Path, rows: list[dict], fields):
    """Write plain-text report in a deterministic UTF-8 CSV format."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_one(source_file: str, *, score_dir: Path, notes_dir: Path,
                output_dir: Path, config: dict, pt):
    """Validate, then save the selected score's alternative MIDI and notes."""
    source = score_dir / source_file
    if not source.is_file():
        raise FileNotFoundError(f"Original MIDI not found: {source}")
    safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", source.stem)[:130]
    baseline_path = notes_dir / (safe_base + ".csv")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline note CSV missing: {baseline_path}")
    baseline = read_baseline(baseline_path, source_file)

    if source_file == MOZART_FILENAME:
        variant = MOZART_VARIANT
        adjusted_midi, stats, raw_audit = adjust_mozart_midi(source)
        max_ticks = 4
        expected_parts = [1, 2]
        for change in raw_audit:
            change["extension_ticks"] = 4
            change["target_onset_tick"] = ""
            change["correction_rule"] = "mozart_duration_4_ticks_short_of_32_grid"
    elif source_file in SONATA_FILES:
        variant = SONATA_VARIANT
        adjusted_midi, stats, raw_audit = adjust_sonata_midi(source)
        max_ticks = SONATA_MAX_EXTENSION_TICKS
        expected_parts = [1, 2, 3]
    else:
        raise ValueError(f"Not a supported experimental MIDI file: {source_file}")

    if not raw_audit:
        raise ValueError(f"No offset adjustments found in {source_file}; "
                         "no alternative exported")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = pt.load_score_midi(adjusted_midi, quantization_unit=1)
    parts = extract.get_parts(score)
    selected, selection_reason, selection_status = extract.get_selection(
        source_file, parts, config)
    if selected != expected_parts:
        raise ValueError(f"Unexpected selected Partitura parts in {source_file}: "
                         f"{selected} (expected {expected_parts})")
    adjusted_rows = []
    for ix in selected:
        adjusted_rows.extend(extract.extract_part(
            parts[ix-1], ix, source_file, variant, pt))
    adjusted_rows = extract.sorted_rows(adjusted_rows)
    changes = compare_to_baseline(baseline, adjusted_rows,
                                  stats["ticks_per_quarter"], max_ticks)
    if not changes:
        raise ValueError("Raw MIDI changed, but no selected exported offsets changed")
    if len(changes) > len(raw_audit):
        raise ValueError("More exported offsets changed than raw MIDI events")
    # Every reimported offset change should be traceable to a changed source
    # MIDI note with the same MIDI pitch, onset and before/after offset.
    expected_changes = Counter((int(a["pitch_midi"]), int(a["onset_tick"]),
                                int(a["original_offset_tick"]),
                                int(a["corrected_offset_tick"]))
                               for a in raw_audit)
    for group_key, orig_tick, adj_tick, _ in changes:
        lookup = (group_key[1], round(group_key[2] * stats["ticks_per_quarter"]),
                  orig_tick, adj_tick)
        # This is a check, not a required one-to-one count: raw notes later
        # merged into a Partitura tie group need not appear as standalone rows.
        if expected_changes[lookup] == 0:
            raise ValueError(f"Unexplained extracted offset change: {lookup}")
        expected_changes[lookup] -= 1

    state_before = _state_diagnostics(baseline)
    state_after = _state_diagnostics(adjusted_rows)
    if state_after["state_count"] > state_before["state_count"]:
        raise ValueError("Unexpected increase in score-state count after extension")
    if state_after["rest_count"] > state_before["rest_count"]:
        raise ValueError("Unexpected increase in rest-state count after extension")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_base + "." + variant
    midi_out = output_dir / (stem + ".mid")
    csv_out, npz_out = extract.write_outputs(adjusted_rows, output_dir / stem)
    adjusted_midi.save(filename=str(midi_out))
    ordinary = sum(not row["is_grace"] for row in adjusted_rows)
    grace = len(adjusted_rows) - ordinary
    audit_path = output_dir / (stem + ".audit.csv")
    audit_rows = [{"source_file": source_file, "variant": variant, **a}
                  for a in raw_audit]
    _atomic_csv(audit_path, audit_rows, _audit_columns())
    extraction_row = {
        "source_file": source_file, "variant": variant, "status": "OK",
        "selected_parts": ";".join(map(str, selected)),
        "selection_status": selection_status, "selection_reason": selection_reason,
        "notes": ordinary, "grace_notes": grace,
        "tied_groups": sum(row["tied_segments"] > 1 for row in adjusted_rows),
        "missing_measure": sum(not row["measure_number"] for row in adjusted_rows),
        "missing_staff": sum(not row["staff"] for row in adjusted_rows),
        "csv": str(csv_out), "npz": str(npz_out), "error": "",
        "parser_warning_count": len(caught),
    }
    summary = {
        "source_file": source_file, "variant": variant, "status": "OK",
        "ticks_per_quarter": stats["ticks_per_quarter"],
        "corrected_raw_note_offs": len(raw_audit),
        "changed_exported_note_offsets": len(changes),
        "baseline_exported_notes": len(baseline),
        "adjusted_exported_notes": len(adjusted_rows),
        "max_extension_ticks": max_ticks,
        "min_applied_extension_ticks": min(int(a["extension_ticks"]) for a in raw_audit),
        "max_applied_extension_ticks": max(int(a["extension_ticks"]) for a in raw_audit),
        "original_states": state_before["state_count"],
        "adjusted_states": state_after["state_count"],
        "original_rests": state_before["rest_count"],
        "adjusted_rests": state_after["rest_count"],
        "original_short_states": state_before["short_count"],
        "adjusted_short_states": state_after["short_count"],
        "raw_pair_count": stats.get("source_pairs", ""),
        "unmatched_raw_note_on": stats.get("unmatched_on", ""),
        "unmatched_raw_note_off": stats.get("unmatched_off", ""),
        "sonata_candidates": stats.get("candidates", ""),
        "sonata_skipped_intervening_off": stats.get("skipped_intervening_off", ""),
        "sonata_skipped_ambiguous": stats.get("skipped_ambiguous", ""),
        "mozart_skipped_overlap": stats.get("skipped_overlap", ""),
        "audit": str(audit_path), "adjusted_midi": str(midi_out),
        "csv": str(csv_out), "npz": str(npz_out), "error": "",
    }
    return extraction_row, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--notes-dir", type=Path, default=DEFAULT_NOTES_DIR,
                        help="Original validated note CSV directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Separate output folder for experimental variants")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--file", choices=SOURCES,
                        help="Process only this MIDI score (default: all four)")
    parser.add_argument("--strict", action="store_true",
                        help="Stop immediately if any file fails")
    args = parser.parse_args(argv)
    try:
        import partitura as pt
        import mido  # noqa: F401
    except ImportError as exc:
        parser.error(f"Missing required dependency: {exc}")
    score_dir, notes_dir, output_dir = (args.score_dir.resolve(),
                                         args.notes_dir.resolve(),
                                         args.output_dir.resolve())
    if output_dir == notes_dir or output_dir == score_dir or score_dir in output_dir.parents:
        parser.error("Experimental output must be outside the baseline and source score directories")
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    targets = (args.file,) if args.file else SOURCES
    output_dir.mkdir(parents=True, exist_ok=True)
    export_rows = []
    report_rows = []
    errors = 0
    for name in targets:
        try:
            export_row, report_row = process_one(
                name, score_dir=score_dir, notes_dir=notes_dir,
                output_dir=output_dir, config=config, pt=pt)
            export_rows.append(export_row)
            report_rows.append(report_row)
            print(f"[OK] {name}: {report_row['corrected_raw_note_offs']} raw note-offs "
                  f"extended; {report_row['changed_exported_note_offsets']} exported "
                  f"notes changed; {report_row['original_states']} -> "
                  f"{report_row['adjusted_states']} score states")
        except Exception as exc:
            errors += 1
            variant = MOZART_VARIANT if name == MOZART_FILENAME else SONATA_VARIANT
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failure = {"source_file": name, "variant": variant, "status": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}"}
            export_rows.append(failure)
            report_rows.append(failure.copy())
            if args.strict:
                raise
    extraction_fields = ["source_file", "variant", "status", "selected_parts",
                         "selection_status", "selection_reason", "notes", "grace_notes",
                         "tied_groups", "missing_measure", "missing_staff", "csv", "npz",
                         "error", "parser_warning_count"]
    comparison_fields = [
        "source_file", "variant", "status", "ticks_per_quarter",
        "corrected_raw_note_offs", "changed_exported_note_offsets",
        "baseline_exported_notes", "adjusted_exported_notes",
        "max_extension_ticks", "min_applied_extension_ticks",
        "max_applied_extension_ticks", "original_states", "adjusted_states",
        "original_rests", "adjusted_rests", "original_short_states",
        "adjusted_short_states", "raw_pair_count", "unmatched_raw_note_on",
        "unmatched_raw_note_off", "sonata_candidates",
        "sonata_skipped_intervening_off", "sonata_skipped_ambiguous",
        "mozart_skipped_overlap", "audit", "adjusted_midi", "csv", "npz",
        "error",
    ]
    _atomic_csv(output_dir / "extraction_summary.csv", export_rows, extraction_fields)
    _atomic_csv(output_dir / "timing_correction_summary.csv", report_rows, comparison_fields)
    print(f"\nSummary: {len(targets) - errors}/{len(targets)} experimental MIDI variants made")
    print(f"Reports: {output_dir / 'timing_correction_summary.csv'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
