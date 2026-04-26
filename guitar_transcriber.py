"""
MonophonicGuitarTabber – ComfyUI custom node
Detects monophonic pitches in an audio clip and renders an ASCII guitar tab.
"""

import logging
import os

import librosa
import numpy as np
import scipy.signal
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guitar tuning definitions
# Strings are ordered low → high.  open_midi gives the open-string MIDI note.
#   E2=40  A2=45  D3=50  G3=55  B3=59  E4=64
# ---------------------------------------------------------------------------
TUNINGS = {
    "Standard E": {
        "strings": ["E", "A", "D", "G", "B", "e"],
        "open_midi": [40, 45, 50, 55, 59, 64],
    },
    "Drop D": {
        "strings": ["D", "A", "D", "G", "B", "e"],
        "open_midi": [38, 45, 50, 55, 59, 64],
    },
    "DADGAD": {
        "strings": ["D", "A", "D", "G", "A", "D"],
        "open_midi": [38, 45, 50, 55, 57, 62],
    },
}

MAX_FRET = 12                 # highest fret considered for mapping
MAX_MEDIAN_KERNEL_DELTA = 30  # simplicity=1.0 → kernel size 1 + MAX_MEDIAN_KERNEL_DELTA


# ---------------------------------------------------------------------------
# Helper functions (module-level so they can be unit-tested independently)
# ---------------------------------------------------------------------------

def _filter_short_notes(midi_notes: np.ndarray, min_hops: int) -> np.ndarray:
    """Zero out note runs that are shorter than *min_hops* frames."""
    result = midi_notes.copy()
    i = 0
    while i < len(result):
        if result[i] != 0:
            j = i
            current = result[i]
            while j < len(result) and result[j] == current:
                j += 1
            if (j - i) < min_hops:
                result[i:j] = 0
            i = j
        else:
            i += 1
    return result


def _to_mono_float32(waveform) -> np.ndarray:
    """
    Convert ComfyUI AUDIO waveform input to a contiguous mono float32 numpy array.

    Expected waveform shapes:
      - torch.Tensor(B, C, T)
      - torch.Tensor(C, T)
      - torch.Tensor(T,)
      - numpy equivalents of the above
    """
    if isinstance(waveform, torch.Tensor):
        x = waveform.detach().to("cpu", dtype=torch.float32)
        if x.ndim == 3:
            # (B, C, T) -> mix batch and channels to mono
            x = x.mean(dim=0).mean(dim=0)
        elif x.ndim == 2:
            # (C, T) -> average channels
            x = x.mean(dim=0)
        elif x.ndim != 1:
            raise ValueError(f"Unsupported tensor waveform shape: {tuple(x.shape)}")
        return np.ascontiguousarray(x.numpy(), dtype=np.float32)

    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 3:
        x = x.mean(axis=0).mean(axis=0)
    elif x.ndim == 2:
        x = x.mean(axis=0)
    elif x.ndim != 1:
        raise ValueError(f"Unsupported ndarray waveform shape: {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)


def _select_hop_length(num_samples: int, sample_rate: int) -> int:
    """
    Choose hop length adaptively so long recordings remain practical.
    """
    duration_s = num_samples / sample_rate
    if duration_s <= 60:
        return 512
    if duration_s <= 180:
        return 1024
    if duration_s <= 600:
        return 2048
    return 4096


def _format_memory_report(
    audio_np: np.ndarray,
    f0: np.ndarray,
    f0_smooth: np.ndarray,
    midi_notes: np.ndarray,
    hop_length: int,
    sample_rate: int,
) -> str:
    """
    Return a concise, human-readable report of memory usage and processing shape.
    """
    duration_s = len(audio_np) / sample_rate
    total_bytes = audio_np.nbytes + f0.nbytes + f0_smooth.nbytes + midi_notes.nbytes
    report = [
        "[Memory Usage]",
        f"duration_sec: {duration_s:.2f}",
        f"hop_length: {hop_length}",
        f"audio_mb: {audio_np.nbytes / (1024 * 1024):.2f}",
        f"f0_mb: {f0.nbytes / (1024 * 1024):.2f}",
        f"f0_smooth_mb: {f0_smooth.nbytes / (1024 * 1024):.2f}",
        f"midi_mb: {midi_notes.nbytes / (1024 * 1024):.2f}",
        f"tracked_total_mb: {total_bytes / (1024 * 1024):.2f}",
    ]
    available_mem_bytes = _get_available_memory_bytes()
    if available_mem_bytes is not None:
        report.append(f"available_system_mb: {available_mem_bytes / (1024 * 1024):.2f}")
    return "\n".join(report)


def _get_available_memory_bytes():
    """
    Best-effort system available memory lookup.
    """
    try:
        import psutil  # optional dependency

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Linux fallback
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass

    # POSIX fallback (may not be available on all systems)
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return int(pages * page_size)
    except Exception:
        pass

    return None


def _serialize_midi_data(midi_notes: np.ndarray, hop_length: int, sample_rate: int) -> str:
    """
    Serialize MIDI contour as event runs:
    start_sec,duration_sec,midi
    """
    if midi_notes.size == 0:
        return ""

    rows = ["start_sec,duration_sec,midi"]
    start = 0
    current = int(midi_notes[0])
    for idx in range(1, len(midi_notes) + 1):
        at_end = idx == len(midi_notes)
        value = None if at_end else int(midi_notes[idx])
        if at_end or value != current:
            duration_hops = idx - start
            start_sec = (start * hop_length) / sample_rate
            duration_sec = (duration_hops * hop_length) / sample_rate
            rows.append(f"{start_sec:.4f},{duration_sec:.4f},{current}")
            if not at_end:
                start = idx
                current = value
    return "\n".join(rows)


def _snap_to_grid(
    midi_notes: np.ndarray,
    simplicity: float,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    """
    Snap the note sequence to a rhythmic grid.

    High simplicity → coarser grid (quarter notes).
    Low  simplicity → fine grid (16th notes).
    Assumes 120 BPM as a neutral reference tempo.
    """
    bpm = 120.0
    beat_duration = 60.0 / bpm  # seconds per beat

    if simplicity > 0.6:
        grid_division = 4   # quarter notes
    elif simplicity > 0.3:
        grid_division = 8   # eighth notes
    else:
        grid_division = 16  # sixteenth notes

    hop_duration = hop_length / sample_rate
    grid_hops = max(1, int((beat_duration / grid_division) / hop_duration))

    result = np.zeros_like(midi_notes)
    for start in range(0, len(midi_notes), grid_hops):
        end = min(start + grid_hops, len(midi_notes))
        cell = midi_notes[start:end]
        voiced = cell[cell != 0]
        if len(voiced) > 0:
            unique, counts = np.unique(voiced, return_counts=True)
            result[start:end] = unique[np.argmax(counts)]
    return result


def _midi_to_string_fret(midi_note: int, tuning_name: str):
    """
    Return *(string_index, fret)* for the lowest-fret position on the guitar,
    or *(None, None)* if the note cannot be played within MAX_FRET.

    String indices follow the TUNINGS definition order (0 = lowest string).
    """
    open_midis = TUNINGS[tuning_name]["open_midi"]
    best_string = None
    best_fret = None
    for idx, open_midi in enumerate(open_midis):
        fret = midi_note - open_midi
        if 0 <= fret <= MAX_FRET:
            if best_fret is None or fret < best_fret:
                best_fret = fret
                best_string = idx
    return best_string, best_fret


def _generate_tab(midi_notes: np.ndarray, tuning_name: str) -> str:
    """
    Convert a frame-level MIDI note array into a human-readable ASCII guitar tab.

    Each unique note event becomes one column.  Up to 32 events are rendered;
    longer sequences are truncated so the output stays readable.
    """
    string_names = TUNINGS[tuning_name]["strings"]
    num_strings = len(string_names)

    # Collapse consecutive identical frames into single events.
    events = []
    prev = -1
    for note in midi_notes:
        n = int(note)
        if n != prev:
            events.append(n)
            prev = n

    # Emit an empty tab when nothing was detected.
    if not events or all(e == 0 for e in events):
        lines = [f"{name}|---|" for name in reversed(string_names)]
        return "\n".join(lines)

    # Trim to 32 note events for readability.
    MAX_EVENTS = 32
    events = events[:MAX_EVENTS]

    # Build a list of columns; each column is a list indexed by string.
    # col[i] == None  → rest on that string
    # col[i] == str   → fret number
    columns = []
    for note in events:
        col = [None] * num_strings
        if note > 0:
            str_idx, fret = _midi_to_string_fret(note, tuning_name)
            if str_idx is not None:
                col[str_idx] = str(fret)
        columns.append(col)

    # Determine the display width of each column (max fret digit count, min 1).
    col_widths = []
    for col in columns:
        w = max(
            (len(v) for v in col if v is not None),
            default=1,
        )
        col_widths.append(w)

    # Render lines from highest string (e) down to lowest (E).
    lines = []
    for str_idx in range(num_strings - 1, -1, -1):
        name = string_names[str_idx]
        parts = [f"{name}|"]
        for col, width in zip(columns, col_widths):
            cell = col[str_idx]
            if cell is not None:
                parts.append(cell.ljust(width, "-") + "-")
            else:
                parts.append("-" * (width + 1))
        parts.append("|")
        lines.append("".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class MonophonicGuitarTabber:
    """
    ComfyUI node that transcribes a monophonic audio clip to a guitar tab.

    Inputs
    ------
    audio      : ComfyUI AUDIO type  (waveform tensor + sample_rate)
    simplicity : 0.0 → detailed tab,  1.0 → highly simplified tab
    tuning     : guitar tuning preset
    threshold  : RMS silence floor – samples below this level are zeroed

    Outputs
    -------
    TAB_TEXT   : multi-line ASCII guitar tab string
    UI_DISPLAY : tab + memory usage summary, for display in a Text widget
    MEMORY_USAGE: memory usage summary text
    MIDI_DATA  : serialized MIDI event runs (CSV-like text)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "simplicity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "tuning": (list(TUNINGS.keys()),),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("TAB_TEXT", "UI_DISPLAY", "MEMORY_USAGE", "MIDI_DATA")
    FUNCTION = "transcribe"
    CATEGORY = "audio"
    OUTPUT_NODE = True

    def transcribe(
        self,
        audio: dict,
        simplicity: float,
        tuning: str,
        threshold: float,
    ):
        # ------------------------------------------------------------------
        # 1.  Extract a mono float32 numpy array from the ComfyUI AUDIO dict.
        #     ComfyUI AUDIO format: {"waveform": Tensor(B, C, T), "sample_rate": int}
        # ------------------------------------------------------------------
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))

        if waveform is None:
            raise ValueError("audio['waveform'] is required.")
        if sample_rate <= 0:
            raise ValueError("audio['sample_rate'] must be a positive integer.")

        audio_np = _to_mono_float32(waveform)
        if audio_np.size == 0:
            raise ValueError("Audio waveform is empty.")

        # ------------------------------------------------------------------
        # 2.  Silence gate.
        # ------------------------------------------------------------------
        audio_np[np.abs(audio_np) < threshold] = 0.0

        # ------------------------------------------------------------------
        # 3.  Fundamental-frequency estimation (pYIN).
        #     Uses adaptive hop_length for long-form audio.
        # ------------------------------------------------------------------
        hop_length = _select_hop_length(len(audio_np), sample_rate)
        f0, voiced_flag, _voiced_probs = librosa.pyin(
            audio_np,
            fmin=float(librosa.note_to_hz("E2")),
            fmax=float(librosa.note_to_hz("E5")),
            sr=sample_rate,
            hop_length=hop_length,
        )

        # pYIN returns NaN for unvoiced frames; replace with 0.
        f0 = np.where(np.isnan(f0), 0.0, f0)

        # ------------------------------------------------------------------
        # 4.  Simplicity-based median filter on the f0 contour.
        #     simplicity 0 → kernel 1 (identity),  1 → kernel 1 + MAX_MEDIAN_KERNEL_DELTA.
        # ------------------------------------------------------------------
        kernel_size = max(1, int(1 + simplicity * MAX_MEDIAN_KERNEL_DELTA))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            f0_smooth = scipy.signal.medfilt(f0, kernel_size=kernel_size)
        else:
            f0_smooth = f0.copy()

        # Re-apply voiced mask so that smoothing doesn't bleed into silences.
        f0_smooth[~voiced_flag] = 0.0

        # ------------------------------------------------------------------
        # 5.  Convert f0 → MIDI note numbers.
        #     MIDI = 12 · log₂(f / 440) + 69
        # ------------------------------------------------------------------
        midi_notes = np.zeros(len(f0_smooth), dtype=int)
        voiced_mask = f0_smooth > 0
        if voiced_mask.any():
            midi_notes[voiced_mask] = np.round(
                12.0 * np.log2(f0_smooth[voiced_mask] / 440.0) + 69
            ).astype(int)

        # ------------------------------------------------------------------
        # 6.  Simplicity > 0.8 → drop notes shorter than 200 ms.
        # ------------------------------------------------------------------
        if simplicity > 0.8:
            hop_duration_s = hop_length / sample_rate
            min_hops = max(1, int(0.200 / hop_duration_s))
            midi_notes = _filter_short_notes(midi_notes, min_hops)

        # ------------------------------------------------------------------
        # 7.  Snap to rhythmic grid.
        # ------------------------------------------------------------------
        midi_notes = _snap_to_grid(midi_notes, simplicity, sample_rate, hop_length)

        # ------------------------------------------------------------------
        # 8.  Render ASCII tab + memory report + MIDI data.
        # ------------------------------------------------------------------
        tab_text = _generate_tab(midi_notes, tuning)
        memory_text = _format_memory_report(
            audio_np=audio_np,
            f0=f0,
            f0_smooth=f0_smooth,
            midi_notes=midi_notes,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )
        midi_data_text = _serialize_midi_data(midi_notes, hop_length, sample_rate)
        ui_display = f"{tab_text}\n\n{memory_text}"

        return (tab_text, ui_display, memory_text, midi_data_text)
