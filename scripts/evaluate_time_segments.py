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

        row = {
            "segment_index": i,
            "t_start": i * SEGMENT_SECONDS,
            "t_end": (i + 1) * SEGMENT_SECONDS,
            "mean": float(segment.mean()),
            "std": float(segment.std()),
        }

        for predictor in PREDICTORS:
            payload, stats = codec.compress(segment, block_size=BLOCK_SIZE, predictors=(predictor,))
            restored = codec.decompress(payload)
            if not np.array_equal(segment, restored):
                raise AssertionError(f"roundtrip mismatch at segment {i}, predictor={predictor}")
            row[f"{predictor}_ratio"] = stats.ratio

        rows.append(row)
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

    fig, ax_ratio = plt.subplots(figsize=(12, 4))

    for predictor in PREDICTORS:
        ratio = [r[f"{predictor}_ratio"] for r in rows]
        ax_ratio.plot(t, ratio, marker="o", markersize=3, label=predictor)

    ax_ratio.set_xlabel("Time [s]")
    ax_ratio.set_ylabel("compression ratio")
    ax_ratio.set_title(f"compression ratio per {SEGMENT_SECONDS:g}s segment (60 segments)")
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_data()
    print(f"loaded {data.size:,} samples from {DATA_PATH}")

    rows = evaluate_segments(data)

    stds = [r["std"] for r in rows]
    print(f"segments: {len(rows)} x {SEGMENT_SECONDS:g}s")
    for predictor in PREDICTORS:
        ratios = [r[f"{predictor}_ratio"] for r in rows]
        correlation = float(np.corrcoef(ratios, stds)[0, 1])
        print(
            f"{predictor:8s} ratio: min={min(ratios):.3f} max={max(ratios):.3f} "
            f"mean={np.mean(ratios):.3f}  correlation(ratio, std)={correlation:.3f}"
        )

    write_csv(rows, RESULTS_DIR / "time_segments.csv")
    plot_results(rows, RESULTS_DIR / "ratio_vs_time.png")
    print(f"wrote {RESULTS_DIR / 'time_segments.csv'}")
    print(f"wrote {RESULTS_DIR / 'ratio_vs_time.png'}")


if __name__ == "__main__":
    main()
