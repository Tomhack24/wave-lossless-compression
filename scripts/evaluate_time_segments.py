import bz2
import csv
import os
import sys
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "wave_lossless_compression_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wave_lossless_compression import codec

DATA_PATH = Path("data/wave_2026.dat")
RESULTS_DIR = Path("results/time_segment_evaluation")

SAMPLE_RATE = 65536
SEGMENT_SECONDS = 1.0
SEGMENT_SIZE = int(SAMPLE_RATE * SEGMENT_SECONDS)
BLOCK_SIZE = 4096
PREDICTORS = ("raw", "diff1", "linear2")


def load_data() -> np.ndarray:
    with open(DATA_PATH, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8)


def evaluate_segments(data: np.ndarray) -> list[dict]:
    n_segments = data.size // SEGMENT_SIZE
    rows = []
    for i in range(n_segments):
        segment = data[i * SEGMENT_SIZE : (i + 1) * SEGMENT_SIZE]

        payload, stats = codec.compress(segment, block_size=BLOCK_SIZE, predictors=PREDICTORS)
        restored = codec.decompress(payload)
        if not np.array_equal(segment, restored):
            raise AssertionError(f"roundtrip mismatch at segment {i}")

        bz2_bytes = len(bz2.compress(segment.tobytes(), compresslevel=9))

        rows.append(
            {
                "segment_index": i,
                "t_start": i * SEGMENT_SECONDS,
                "t_end": (i + 1) * SEGMENT_SECONDS,
                "mean": float(segment.mean()),
                "std": float(segment.std()),
                "wlc_ratio": stats.ratio,
                "wlc_bits_per_sample": stats.bits_per_sample,
                "bz2_ratio": segment.size / bz2_bytes,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict], path: Path) -> None:
    t = [r["t_start"] for r in rows]
    wlc_ratio = [r["wlc_ratio"] for r in rows]
    bz2_ratio = [r["bz2_ratio"] for r in rows]
    std = [r["std"] for r in rows]

    fig, (ax_ratio, ax_std) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax_ratio.plot(t, wlc_ratio, marker="o", markersize=3, label="wlc/adaptive (proposed)", color="#4C72B0")
    ax_ratio.plot(t, bz2_ratio, marker="o", markersize=3, label="bz2 (general-purpose)", color="#888888")
    ax_ratio.set_ylabel("compression ratio")
    ax_ratio.set_title(f"compression ratio per {SEGMENT_SECONDS:g}s segment (60 segments)")
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3)

    ax_std.fill_between(t, std, color="steelblue", alpha=0.7)
    ax_std.set_xlabel("Time [s]")
    ax_std.set_ylabel("amplitude std [uint8]")
    ax_std.set_title("segment amplitude std (activity level)")
    ax_std.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_data()
    print(f"loaded {data.size:,} samples from {DATA_PATH}")

    rows = evaluate_segments(data)

    ratios = [r["wlc_ratio"] for r in rows]
    stds = [r["std"] for r in rows]
    correlation = float(np.corrcoef(ratios, stds)[0, 1])

    print(f"segments: {len(rows)} x {SEGMENT_SECONDS:g}s")
    print(f"wlc/adaptive ratio: min={min(ratios):.3f} max={max(ratios):.3f} mean={np.mean(ratios):.3f}")
    print(f"correlation(ratio, std) = {correlation:.3f}")

    write_csv(rows, RESULTS_DIR / "time_segments.csv")
    plot_results(rows, RESULTS_DIR / "ratio_vs_time.png")
    print(f"wrote {RESULTS_DIR / 'time_segments.csv'}")
    print(f"wrote {RESULTS_DIR / 'ratio_vs_time.png'}")


if __name__ == "__main__":
    main()
