"""
MonophonicGuitarTabber – ComfyUI custom node
Detects monophonic pitches in an audio clip and renders an ASCII guitar tab.
"""

import logging

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
MAX_AUDIO_SECONDS = 60        # safety limit (seconds)
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
    UI_DISPLAY : same string, for display in a Text widget
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("TAB_TEXT", "UI_DISPLAY")
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
        waveform = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        if isinstance(waveform, torch.Tensor):
            # (B, C, T) → squeeze batch → average channels → (T,)
            audio_np = (
                waveform.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
            )
        else:
            audio_np = (
                np.asarray(waveform, dtype=np.float32).squeeze(0).mean(axis=0)
            )

        # ------------------------------------------------------------------
        # 2.  Safety / memory check – truncate if longer than 60 seconds.
        # ------------------------------------------------------------------
        max_samples = int(MAX_AUDIO_SECONDS * sample_rate)
        if len(audio_np) > max_samples:
            logger.warning(
                "[MonophonicGuitarTabber] Audio exceeds %ds (%.1fs). "
                "Truncating to prevent RAM/VRAM spikes.",
                MAX_AUDIO_SECONDS,
                len(audio_np) / sample_rate,
            )
            audio_np = audio_np[:max_samples]

        # ------------------------------------------------------------------
        # 3.  Silence gate.
        # ------------------------------------------------------------------
        audio_np[np.abs(audio_np) < threshold] = 0.0

        # ------------------------------------------------------------------
        # 4.  Fundamental-frequency estimation (pYIN).
        # ------------------------------------------------------------------
        hop_length = 512
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
        # 5.  Simplicity-based median filter on the f0 contour.
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
        # 6.  Convert f0 → MIDI note numbers.
        #     MIDI = 12 · log₂(f / 440) + 69
        # ------------------------------------------------------------------
        midi_notes = np.zeros(len(f0_smooth), dtype=int)
        voiced_mask = f0_smooth > 0
        if voiced_mask.any():
            midi_notes[voiced_mask] = np.round(
                12.0 * np.log2(f0_smooth[voiced_mask] / 440.0) + 69
            ).astype(int)

        # ------------------------------------------------------------------
        # 7.  Simplicity > 0.8 → drop notes shorter than 200 ms.
        # ------------------------------------------------------------------
        if simplicity > 0.8:
            hop_duration_s = hop_length / sample_rate
            min_hops = max(1, int(0.200 / hop_duration_s))
            midi_notes = _filter_short_notes(midi_notes, min_hops)

        # ------------------------------------------------------------------
        # 8.  Snap to rhythmic grid.
        # ------------------------------------------------------------------
        midi_notes = _snap_to_grid(midi_notes, simplicity, sample_rate, hop_length)

        # ------------------------------------------------------------------
        # 9.  Render ASCII tab.
        # ------------------------------------------------------------------
        tab_text = _generate_tab(midi_notes, tuning)

        return (tab_text, tab_text)
