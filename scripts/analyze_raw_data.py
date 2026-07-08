import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "wave_lossless_compression_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 65536


def load_uint8_dat(input_path: Path) -> np.ndarray:
    with open(input_path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8)


def basic_stats(data: np.ndarray) -> dict[str, float]:
    return {
        "samples": float(data.size),
        "duration_seconds": data.size / SAMPLE_RATE,
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def downsample_min_max(data: np.ndarray, target_points: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_size = max(1, data.size // target_points)
    usable = data[: (data.size // block_size) * block_size]
    blocks = usable.reshape(-1, block_size)

    time = np.arange(blocks.shape[0]) * block_size / SAMPLE_RATE
    return time, blocks.min(axis=1), blocks.max(axis=1)


def fft_spectrum(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = data.astype(np.float32) - data.mean()
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    frequency = np.fft.rfftfreq(signal.size, d=1 / SAMPLE_RATE)
    amplitude = np.abs(spectrum)
    return frequency[1:], amplitude[1:]


def save_waveform(data: np.ndarray, output_path: Path) -> None:
    time, low, high = downsample_min_max(data)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(time, low, high, color="steelblue", linewidth=0, alpha=0.85)
    ax.set_title("Raw waveform envelope")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [uint8]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_histogram(data: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=np.arange(257), color="steelblue", edgecolor="none")
    ax.axvline(data.mean(), color="crimson", linewidth=1.5, label=f"mean = {data.mean():.2f}")
    ax.set_title("Value distribution")
    ax.set_xlabel("Amplitude [uint8]")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_fft(data: np.ndarray, output_path: Path) -> None:
    frequency, amplitude = fft_spectrum(data)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frequency, amplitude, color="darkslateblue", linewidth=0.5)
    ax.set_title("FFT spectrum")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_spectrogram(data: np.ndarray, output_path: Path, max_frequency: int | None = None) -> None:
    signal = data.astype(np.float32) - data.mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    spectrogram, frequency, time, image = ax.specgram(
        signal,
        NFFT=4096,
        Fs=SAMPLE_RATE,
        noverlap=3072,
        cmap="magma",
        scale="dB",
    )
    if max_frequency is not None:
        ax.set_ylim(0, max_frequency)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(image, ax=ax, label="Power [dB]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_overview(data: np.ndarray, stats: dict[str, float], output_path: Path) -> None:
    time, low, high = downsample_min_max(data)
    frequency, amplitude = fft_spectrum(data)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].fill_between(time, low, high, color="steelblue", linewidth=0, alpha=0.85)
    axes[0, 0].set_title("Waveform envelope")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude [uint8]")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(data, bins=np.arange(257), color="steelblue", edgecolor="none")
    axes[0, 1].axvline(stats["mean"], color="crimson", linewidth=1.5)
    axes[0, 1].set_title("Value distribution")
    axes[0, 1].set_xlabel("Amplitude [uint8]")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].plot(frequency, amplitude, color="darkslateblue", linewidth=0.5)
    axes[1, 0].set_title("FFT spectrum")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    summary = "\n".join(
        [
            "Summary",
            f"samples: {int(stats['samples'])}",
            f"duration: {stats['duration_seconds']:.2f} s",
            f"sample rate: {SAMPLE_RATE} Hz",
            f"mean: {stats['mean']:.2f}",
            f"std: {stats['std']:.2f}",
            f"min: {int(stats['min'])}",
            f"max: {int(stats['max'])}",
        ]
    )
    axes[1, 1].text(0.05, 0.95, summary, va="top", family="monospace", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary_text(stats: dict[str, float], output_path: Path) -> None:
    output_path.write_text(
        "\n".join(
            [
                f"samples: {int(stats['samples'])}",
                f"duration_seconds: {stats['duration_seconds']:.2f}",
                f"sample_rate_hz: {SAMPLE_RATE}",
                f"mean: {stats['mean']:.2f}",
                f"std: {stats['std']:.2f}",
                f"min: {int(stats['min'])}",
                f"max: {int(stats['max'])}",
                "",
            ]
        )
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "wave_2026.dat"
    output_dir = project_root / "results" / "raw_data_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_uint8_dat(input_path)
    stats = basic_stats(data)

    save_overview(data, stats, output_dir / "overview.png")
    save_waveform(data, output_dir / "waveform.png")
    save_histogram(data, output_dir / "histogram.png")
    save_fft(data, output_dir / "fft_spectrum.png")
    save_spectrogram(data, output_dir / "spectrogram_0_2khz.png", max_frequency=2000)
    save_spectrogram(data, output_dir / "spectrogram_full.png")
    save_summary_text(stats, output_dir / "summary.txt")

    print(f"samples: {int(stats['samples'])}")
    print(f"duration: {stats['duration_seconds']:.2f} seconds")
    print(
        f"mean: {stats['mean']:.2f}, std: {stats['std']:.2f}, "
        f"min: {int(stats['min'])}, max: {int(stats['max'])}"
    )
    print(f"saved results to: {output_dir}")


if __name__ == "__main__":
    main()
