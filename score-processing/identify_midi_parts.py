from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import mido
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SCORE_DIR = PROJECT_ROOT / "score"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "midi_part_review"


@dataclass(frozen=True)
class Note:
    onset_tick: int
    offset_tick: int
    pitch: int
    channel: int


@dataclass
class TrackProfile:
    track_index: int  # 0-based Mido track index, NOT Partitura part index
    track_name: str
    notes: list[Note]
    programs: list[str]
    channels: list[int]
    unmatched_note_ons: int
    unmatched_note_offs: int

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def first_onset(self) -> int | None:
        return min((n.onset_tick for n in self.notes), default=None)

    @property
    def last_offset(self) -> int | None:
        return max((n.offset_tick for n in self.notes), default=None)

    @property
    def median_pitch(self) -> float | None:
        return float(np.median([n.pitch for n in self.notes])) if self.notes else None

    @property
    def min_pitch(self) -> int | None:
        return min((n.pitch for n in self.notes), default=None)

    @property
    def max_pitch(self) -> int | None:
        return max((n.pitch for n in self.notes), default=None)


def analyze_track(track: mido.MidiTrack, index: int) -> TrackProfile:
    """Pair note-ons and note-offs without modifying source MIDI events."""
    absolute_tick = 0
    active: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    notes: list[Note] = []
    programs: set[str] = set()
    channels: set[int] = set()
    unmatched_offs = 0

    for msg in track:
        absolute_tick += msg.time
        if msg.is_meta:
            continue
        if hasattr(msg, "channel"):
            channels.add(msg.channel)
        if msg.type == "program_change":
            programs.add(f"ch{msg.channel}:program{msg.program}")
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)].append(absolute_tick)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if active[key]:
                start = active[key].popleft()
                if absolute_tick > start:
                    notes.append(Note(start, absolute_tick, msg.note, msg.channel))
            else:
                unmatched_offs += 1

    unmatched_ons = sum(len(q) for q in active.values())
    return TrackProfile(
        track_index=index,
        track_name=track.name or "(unnamed)",
        notes=sorted(notes, key=lambda n: (n.onset_tick, n.pitch, n.offset_tick)),
        programs=sorted(programs),
        channels=sorted(channels),
        unmatched_note_ons=unmatched_ons,
        unmatched_note_offs=unmatched_offs,
    )


def group_by_entrance(profiles: list[TrackProfile], ticks_per_beat: int):
    """Propose early-vs-late groups only when entrances are clearly separated.

    Heuristic (our engineering choice, NOT a musical truth): at least four
    sounding tracks, at least two on each side of an entrance gap of 16 or
    more quarter notes, and the gap must exceed the within-group spans.
    """
    sounding = sorted(
        (p for p in profiles if p.n_notes),
        key=lambda p: (p.first_onset, p.track_index),
    )
    if len(sounding) < 4:
        return None

    onsets = [p.first_onset / ticks_per_beat for p in sounding]
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    candidates = []
    for i in range(1, len(sounding) - 2 + 1):
        if i < 2 or len(sounding) - i < 2:
            continue
        gap = gaps[i - 1]
        early_span = onsets[i - 1] - onsets[0]
        late_span = onsets[-1] - onsets[i]
        if gap >= 16 and gap > max(early_span, late_span):
            candidates.append((gap, i))
    if not candidates:
        return None
    gap, split = max(candidates)
    return {
        "early_tracks": [p.track_index for p in sounding[:split]],
        "late_tracks": [p.track_index for p in sounding[split:]],
        "gap_quarters": gap,
        "first_late_quarter": onsets[split],
    }


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")[:100]


def to_absolute(track: mido.MidiTrack):
    tick = 0
    for msg in track:
        tick += msg.time
        yield tick, msg


def make_export(source: mido.MidiFile, indices: list[int], output: Path) -> None:
    """Export selected *raw MIDI tracks*, preserving original time and tempo.

    The new conductor track collects tempo/time/key signature meta-events
    from all tracks. Selected tracks retain all their events, including
    program changes and controllers. No event quantization is performed.
    """
    result = mido.MidiFile(type=1, ticks_per_beat=source.ticks_per_beat)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    meta_events = []
    for source_index, track in enumerate(source.tracks):
        for t, msg in to_absolute(track):
            if msg.is_meta and msg.type in {"set_tempo", "time_signature", "key_signature"}:
                # Avoid duplicates that encode the same meta message at same tick.
                meta_events.append((t, source_index, msg))
    meta_events.sort(key=lambda x: (x[0], x[1]))
    prior, seen = 0, set()
    for t, _, msg in meta_events:
        signature = (t, msg.type, str(msg))
        if signature in seen:
            continue
        seen.add(signature)
        conductor.append(msg.copy(time=t - prior))
        prior = t
    result.tracks.append(conductor)
    for idx in indices:
        original = source.tracks[idx]
        # Removing meta events with nonzero delta time would shift events;
        # reconstruct absolute ticks instead to preserve exact note timings.
        selected = mido.MidiTrack()
        prev = 0
        for t, msg in to_absolute(original):
            if msg.is_meta and msg.type in {"set_tempo", "time_signature", "key_signature", "end_of_track"}:
                continue
            selected.append(msg.copy(time=t - prev))
            prev = t
        selected.append(mido.MetaMessage("end_of_track", time=0))
        result.tracks.append(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def piano_roll(profiles: list[TrackProfile], tpq: int, path: Path, xlim=None) -> None:
    musical = [p for p in profiles if p.n_notes]
    if not musical:
        return
    fig, axes = plt.subplots(len(musical), 1, figsize=(13, max(3, 2.4 * len(musical))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, musical):
        segments = [((n.onset_tick / tpq, n.pitch), (n.offset_tick / tpq, n.pitch)) for n in p.notes]
        ax.add_collection(LineCollection(segments, linewidths=1.2, colors="C0"))
        ax.set_ylim(max(0, p.min_pitch - 3), min(127, p.max_pitch + 3))
        ax.set_ylabel(f"Track {p.track_index}\nMIDI pitch")
        ax.set_title(f"Track {p.track_index} — {p.track_name} ({p.n_notes:,} notes)", fontsize=10, loc="left")
        ax.grid(alpha=0.2)
    end_q = max(p.last_offset for p in musical) / tpq
    axes[-1].set_xlim(*(xlim if xlim else (0, max(1, end_q))))
    axes[-1].set_xlabel("Position (quarter notes from MIDI start)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def output_rows(filename: str, profiles: list[TrackProfile], tpq: int):
    rows = []
    for p in profiles:
        if not p.notes:
            continue
        rows.append({
            "file": filename,
            "raw_midi_track_0_based": p.track_index,
            "raw_track_name": p.track_name,
            "channels_0_based": ";".join(map(str, p.channels)),
            "programs_0_based": ";".join(p.programs),
            "notes": p.n_notes,
            "first_quarter": round(p.first_onset / tpq, 4),
            "last_quarter": round(p.last_offset / tpq, 4),
            "pitch_min": p.min_pitch,
            "pitch_max": p.max_pitch,
            "pitch_median": round(p.median_pitch, 2),
            "unmatched_note_ons": p.unmatched_note_ons,
            "unmatched_note_offs": p.unmatched_note_offs,
            "partitura_mapping": "NOT ASSUMED — raw track number may differ from Partitura part number",
        })
    return rows


def inspect_file(path: Path, output_dir: Path):
    src = mido.MidiFile(path)
    if src.type == 2:
        raise ValueError("MIDI type 2 files are asynchronous; track grouping is not comparable")
    tpq = src.ticks_per_beat
    if tpq <= 0:
        raise ValueError("SMPTE-based MIDI time division is not supported by this diagnostic")
    profiles = [analyze_track(tr, i) for i, tr in enumerate(src.tracks)]
    sounding = [p for p in profiles if p.n_notes]
    if not sounding:
        raise ValueError("No complete note pairs found")
    out = output_dir / safe_name(path)
    out.mkdir(parents=True, exist_ok=True)
    piano_roll(profiles, tpq, out / "piano_roll_overview.png")
    piano_roll(profiles, tpq, out / "piano_roll_opening.png", xlim=(0, min(48, max(p.last_offset for p in sounding) / tpq)))
    guess = group_by_entrance(profiles, tpq)
    lines = [f"FILE: {path.name}", f"Ticks per quarter: {tpq}",
             f"Raw MIDI tracks: {len(profiles)}; sounding tracks: {len(sounding)}", "",
             "TRACK EVIDENCE (0-based raw MIDI indices; NOT Partitura part IDs):"]
    for p in sounding:
        lines.append(f"  Track {p.track_index} ({p.track_name}): notes={p.n_notes}, "
                     f"first={p.first_onset/tpq:.3f} q, last={p.last_offset/tpq:.3f} q, "
                     f"pitch={p.min_pitch}..{p.max_pitch}, median={p.median_pitch:.1f}, "
                     f"channels={p.channels}, programs={p.programs or '[no program change]'}, "
                     f"unmatched on/off={p.unmatched_note_ons}/{p.unmatched_note_offs}")
    if any(len(p.channels) > 1 for p in sounding):
        lines.extend(["", "WARNING: Some raw MIDI tracks contain multiple channels; a Partitura part may not map 1:1 to tracks."])
    lines.extend(["", "PROVISIONAL PART-GROUP HYPOTHESIS:"])
    if guess is None:
        lines.extend([
            "  NO AUTOMATIC SOLO/ACCOMPANIMENT DECISION.",
            "  No clear separation between early and late entrances was found.",
            "  Inspect the piano rolls and track metadata; do not discard sparse tracks automatically.",
        ])
    else:
        early, late = guess["early_tracks"], guess["late_tracks"]
        lines.extend([
            f"  Earlier-entering raw tracks: {early}",
            f"  Later-entering raw tracks:   {late}",
            f"  Entrance gap: {guess['gap_quarters']:.2f} quarter notes",
            "  POSSIBLE INTERPRETATION: later tracks could be the solo instrument; earlier tracks could be accompaniment.",
            "  This is a heuristic inference, NOT a verified instrument identification.",
            "  Compare the candidate groups to score or audio evidence before selecting parts for extraction.",
        ])
        piano_roll(profiles, tpq, out / "piano_roll_late_entrance.png",
                   xlim=(max(0, guess["first_late_quarter"] - 8), guess["first_late_quarter"] + 32))
        make_export(src, early, out / "candidate_early_tracks.mid")
        make_export(src, late, out / "candidate_late_tracks.mid")
    # Individual tracks are useful for reviewing or listening, independent of hypothesis.
    for p in sounding:
        make_export(src, [p.track_index], out / f"track_{p.track_index}.mid")
    lines.extend(["", "No note timings, pitches, or source files were changed.",
                  "Standalone MIDI previews preserve the original tempo events."])
    (out / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_rows(path.name, profiles, tpq), {
        "file": path.name,
        "status": "PROVISIONAL_ENTRANCE_GROUPS" if guess else "NO_AUTO_DECISION",
        "earlier_raw_tracks_0_based": ";".join(map(str, guess["early_tracks"])) if guess else "",
        "later_raw_tracks_0_based": ";".join(map(str, guess["late_tracks"])) if guess else "",
        "entrance_gap_quarters": round(guess["gap_quarters"], 3) if guess else "",
        "action": "verify later group against piano score/audio" if guess else "verify all musical tracks",
    }


def csv_write(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--file", type=str, help="Only this MIDI filename or path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    score_dir = args.score_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if args.file:
        candidate = Path(args.file).expanduser()
        if not candidate.is_file():
            candidate = score_dir / args.file
        paths = [candidate]
    else:
        paths = sorted([*score_dir.rglob("*.mid"), *score_dir.rglob("*.midi")])
    if not paths:
        parser.error(f"No MIDI files found in {score_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_guesses = [], []
    for path in paths:
        try:
            rows, guess = inspect_file(path, out_dir)
            all_rows.extend(rows)
            all_guesses.append(guess)
            print(f"[OK] {path.name}: {guess['status']}")
        except Exception as exc:
            print(f"[ERROR] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            all_guesses.append({"file": path.name, "status": "ERROR", "action": f"{type(exc).__name__}: {exc}"})
    csv_write(out_dir / "track_profiles.csv", all_rows,
              ["file", "raw_midi_track_0_based", "raw_track_name", "channels_0_based",
               "programs_0_based", "notes", "first_quarter", "last_quarter", "pitch_min",
               "pitch_max", "pitch_median", "unmatched_note_ons", "unmatched_note_offs", "partitura_mapping"])
    csv_write(out_dir / "group_hypotheses.csv", all_guesses,
              ["file", "status", "earlier_raw_tracks_0_based", "later_raw_tracks_0_based",
               "entrance_gap_quarters", "action"])
    print(f"\nReports saved to: {out_dir}")
    print("Start with group_hypotheses.csv and each file's analysis.txt / piano_roll_overview.png")
    if any(g["status"] == "ERROR" for g in all_guesses):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
