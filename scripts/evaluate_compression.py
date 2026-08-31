import bz2
import csv
import lzma
import os
import sys
import tempfile
import time
import zlib
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "wave_lossless_compression_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wave_lossless_compression import codec, rice

DATA_PATH = Path("data/wave_2026.dat")
RESULTS_DIR = Path("results/compression_evaluation")
REPORT_PATH = Path("report/compression_report.md")

BLOCK_SIZES = [256, 1024, 4096, 16384, 65536]
PREDICTOR_MODES = {
    "raw": ("raw",),
    "diff1": ("diff1",),
    "linear2": ("linear2",),
    "adaptive": ("raw", "diff1", "linear2"),
}


def load_data() -> np.ndarray:
    with open(DATA_PATH, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8)


def shannon_entropy_bits(values: np.ndarray) -> float:
    """Zeroth-order Shannon entropy, in bits per symbol."""
    counts = np.bincount(values.astype(np.int64) - int(values.min()))
    probs = counts[counts > 0] / values.size
    return float(-np.sum(probs * np.log2(probs)))


def evaluate_codec_configs(data: np.ndarray) -> list[dict]:
    rows = []
    for block_size in BLOCK_SIZES:
        for mode_name, predictors in PREDICTOR_MODES.items():
            t0 = time.perf_counter()
            payload, stats = codec.compress(data, block_size=block_size, predictors=predictors)
            t1 = time.perf_counter()
            restored = codec.decompress(payload)
            t2 = time.perf_counter()

            if not np.array_equal(data, restored):
                raise AssertionError(f"roundtrip mismatch: block_size={block_size}, mode={mode_name}")

            rows.append(
                {
                    "method": f"wlc/{mode_name}",
                    "block_size": block_size,
                    "compressed_bytes": stats.compressed_bytes,
                    "ratio": stats.ratio,
                    "bits_per_sample": stats.bits_per_sample,
                    "predictor_counts": dict(stats.predictor_counts),
                    "encode_seconds": t1 - t0,
                    "decode_seconds": t2 - t1,
                }
            )
            print(
                f"wlc/{mode_name:8s} block={block_size:6d}  "
                f"{stats.compressed_bytes:8d} bytes  ratio={stats.ratio:5.3f}  "
                f"{stats.bits_per_sample:5.3f} bit/sample  "
                f"encode={t1 - t0:5.2f}s decode={t2 - t1:5.2f}s"
            )
    return rows


def evaluate_baselines(data: np.ndarray) -> list[dict]:
    raw_bytes = data.tobytes()
    rows = []

    baselines = {
        "zlib(level=9)": lambda b: zlib.compress(b, level=9),
        "bz2(level=9)": lambda b: bz2.compress(b, compresslevel=9),
        "lzma(preset=9|EXTREME)": lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME),
    }
    for name, fn in baselines.items():
        t0 = time.perf_counter()
        compressed = fn(raw_bytes)
        t1 = time.perf_counter()
        rows.append(
            {
                "method": name,
                "block_size": None,
                "compressed_bytes": len(compressed),
                "ratio": data.size / len(compressed),
                "bits_per_sample": len(compressed) * 8 / data.size,
                "predictor_counts": {},
                "encode_seconds": t1 - t0,
                "decode_seconds": float("nan"),
            }
        )
        print(f"{name:24s}{len(compressed):8d} bytes  ratio={data.size / len(compressed):5.3f}")
    return rows


def evaluate_entropy_reference(data: np.ndarray) -> list[dict]:
    from wave_lossless_compression.predictors import diff1_encode, linear2_encode

    rows = []
    for name, residual in [
        ("entropy(raw)", data.astype(np.int16)),
        ("entropy(diff1)", diff1_encode(data)),
        ("entropy(linear2)", linear2_encode(data)),
    ]:
        unsigned = rice.zigzag_encode(residual)
        bits = shannon_entropy_bits(unsigned)
        rows.append(
            {
                "method": name,
                "block_size": None,
                "compressed_bytes": bits * data.size / 8,
                "ratio": 8 / bits,
                "bits_per_sample": bits,
                "predictor_counts": {},
                "encode_seconds": float("nan"),
                "decode_seconds": float("nan"),
            }
        )
        print(f"{name:24s}{bits:5.3f} bit/sample (theoretical, 0th-order entropy)")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "block_size",
        "compressed_bytes",
        "ratio",
        "bits_per_sample",
        "predictor_counts",
        "encode_seconds",
        "decode_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(codec_rows: list[dict], baseline_rows: list[dict], entropy_rows: list[dict], path: Path) -> None:
    fig, (ax_ratio, ax_block) = plt.subplots(1, 2, figsize=(12, 5))

    best_by_mode: dict[str, dict] = {}
    for row in codec_rows:
        mode = row["method"]
        if mode not in best_by_mode or row["ratio"] > best_by_mode[mode]["ratio"]:
            best_by_mode[mode] = row

    labels = list(best_by_mode.keys()) + [r["method"] for r in baseline_rows]
    ratios = [best_by_mode[m]["ratio"] for m in best_by_mode] + [r["ratio"] for r in baseline_rows]
    colors = ["#4C72B0"] * len(best_by_mode) + ["#888888"] * len(baseline_rows)

    ax_ratio.bar(labels, ratios, color=colors)
    ax_ratio.axhline(1.0, color="black", linewidth=0.8)
    ax_ratio.set_ylabel("compression ratio (original / compressed)")
    ax_ratio.set_title("best ratio per method")
    ax_ratio.tick_params(axis="x", rotation=45)
    for tick in ax_ratio.get_xticklabels():
        tick.set_ha("right")

    for mode in PREDICTOR_MODES:
        sizes = [row["block_size"] for row in codec_rows if row["method"] == f"wlc/{mode}"]
        ratios_by_size = [row["ratio"] for row in codec_rows if row["method"] == f"wlc/{mode}"]
        ax_block.plot(sizes, ratios_by_size, marker="o", label=mode)
    ax_block.set_xscale("log", base=2)
    ax_block.set_xlabel("block size (samples)")
    ax_block.set_ylabel("compression ratio")
    ax_block.set_title("wlc: ratio vs block size")
    ax_block.legend()
    ax_block.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(codec_rows: list[dict], baseline_rows: list[dict], entropy_rows: list[dict]) -> str:
    all_rows = codec_rows + baseline_rows + entropy_rows
    best = max((r for r in codec_rows + baseline_rows), key=lambda r: r["ratio"])

    lines = []
    lines.append("# 圧縮手法の比較レポート")
    lines.append("")
    lines.append(f"対象データ: `{DATA_PATH}`（uint8, {os.path.getsize(DATA_PATH):,} バイト）")
    lines.append("")
    lines.append("## 手法")
    lines.append("")
    lines.append(
        "全体の流れは「予測残差への変換（ブロック単位、複数手法から選択可）"
        "→ 残差を Rice 符号でビット列に符号化」の2段階。"
        "実装は `src/wave_lossless_compression/predictors.py`（予測残差）、"
        "`rice.py`（Rice符号）、`codec.py`（ブロック分割・手法選択・全体の圧縮/復元）。"
        "すべての設定について、圧縮したデータを実際に復号して元データと完全に一致すること"
        "（可逆性）を確認済み。"
    )
    lines.append("")
    lines.append("### 予測残差変換（提案手法の入力側）")
    lines.append("")
    lines.append(
        "サンプル列 `x[0], x[1], ...`（各 uint8, 0-255）に対して、次の3種類の可逆変換を用意した。"
        "どれも「実際の値」と「予測値」の差（残差）を計算するだけで、"
        "残差から元の値を厳密に復元できる。"
    )
    lines.append("")
    lines.append(
        "- **raw**: 予測を行わず `residual[i] = x[i]` とする。他手法との比較用のベースライン。\n"
        "- **diff1**（1階差分）: 直前の1点から予測する。`residual[i] = x[i] - x[i-1]`"
        "（先頭のみ `residual[0] = x[0]`）。信号がゆっくり変化するほど残差が0付近に集中する。\n"
        "- **linear2**（2点線形予測）: 直前2点の傾きをそのまま延長して予測する。"
        "`prediction[i] = 2*x[i-1] - x[i-2]`、`residual[i] = x[i] - prediction[i]`"
        "（先頭2点は `raw` と同じ）。信号が直線的に変化する区間では `diff1` より残差が"
        "小さくなりうるが、変化が急な区間ではかえって外れやすい。"
    )
    lines.append("")
    lines.append("### Rice符号（提案手法の符号化側）")
    lines.append("")
    lines.append(
        "残差は負の値も取りうるため、まず zigzag変換で符号なし整数に変換する"
        "（`0, -1, 1, -2, 2, ... → 0, 1, 2, 3, 4, ...`）。"
        "そのうえで、パラメータ `k` を1つ選び、各値 `v` を"
        "「商 `v >> k` を単進符号（1をq個並べて0で終端）」+「余り `v & (2^k - 1)` を"
        "固定長kビット」で表す。`k` が大きいほど固定長部分が増え、小さいほど単進符号部分が"
        "増えるため、値の大きさの分布に応じて最適な `k` が変わる。"
        "本実装ではブロックごとに、候補となる `k`（0〜16）それぞれで符号長"
        "`n*(k+1) + sum(v >> k)` を計算し、最小になる `k` を選んでいる"
        "（`rice.best_k`）。"
    )
    lines.append("")
    lines.append("### ブロック単位の適応選択（`wlc/adaptive`）")
    lines.append("")
    lines.append(
        "固定長ブロック（例: 4096サンプル）ごとに、`raw` / `diff1` / `linear2` それぞれで"
        "残差を計算し、各手法についてRice符号の最適な `k` と符号長を求め、"
        "符号長が最小になる（予測手法, k）の組をそのブロックの符号化方式として採用する。"
        "採用した予測手法（2bit）と `k`（6bit）は1ブロックあたり1バイトのメタデータとして"
        "別途保存し、復号時にどの手法を使ったブロックかを判別する。"
        "`wlc/raw` / `wlc/diff1` / `wlc/linear2` は、この選択を行わず全ブロックで"
        "同じ予測手法に固定した場合（比較用）。"
    )
    lines.append("")
    lines.append(
        "各設定をブロックサイズ "
        + ", ".join(str(b) for b in BLOCK_SIZES)
        + " サンプルで評価した。ブロックが小さいほど局所的な変化に追従しやすい一方、"
        "メタデータ（1ブロック1バイト）や `k` の再推定コストの相対的な割合が増える。"
    )
    lines.append("")
    lines.append("### 比較対象（汎用圧縮）")
    lines.append("")
    lines.append(
        "提案手法が「局所的な相関」をどこまで活かせているかを見るため、"
        "アルゴリズムの異なる汎用圧縮とも比較した（いずれも Python標準ライブラリ、レベル最大）。"
    )
    lines.append("")
    lines.append(
        "- **zlib**（DEFLATE）: LZ77（直近32KB以内の繰り返し部分を過去への参照に置き換える）"
        "+ Huffman符号の組み合わせ。\n"
        "- **bz2**: ブロックソート（BWT）で似た文脈の値を並び替えて集め、"
        "MTF・RLEを経てHuffman符号化する。ブロック内（デフォルト900KB）であれば"
        "離れた位置の繰り返しパターンもまとめて活かせる。\n"
        "- **lzma**（xz形式）: LZ77を大きな辞書サイズ・高精度な範囲符号化（range coder）で"
        "拡張したもの。zlibよりも遠くの繰り返しを見つけやすい。"
    )
    lines.append("")
    lines.append(
        "参考として、0次のシャノンエントロピー"
        "（隣接サンプル間の関係を使わず、値の出現頻度だけから求まる理論的な下限値。"
        "実際の符号化オーバーヘッドは含まない理想値）も併記した。"
    )
    lines.append("")
    lines.append("## 結果")
    lines.append("")
    lines.append("![compression comparison](../results/compression_evaluation/comparison.png)")
    lines.append("")
    lines.append("### 提案手法（ブロック単位の予測選択 + Rice 符号）")
    lines.append("")
    lines.append("| method | block_size | compressed bytes | ratio | bit/sample | encode(s) | decode(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in codec_rows:
        lines.append(
            f"| {row['method']} | {row['block_size']} | {row['compressed_bytes']:,} | "
            f"{row['ratio']:.3f} | {row['bits_per_sample']:.3f} | "
            f"{row['encode_seconds']:.2f} | {row['decode_seconds']:.2f} |"
        )
    lines.append("")
    lines.append("### 汎用圧縮との比較")
    lines.append("")
    lines.append("| method | compressed bytes | ratio | bit/sample |")
    lines.append("|---|---:|---:|---:|")
    for row in baseline_rows:
        lines.append(
            f"| {row['method']} | {row['compressed_bytes']:,} | {row['ratio']:.3f} | {row['bits_per_sample']:.3f} |"
        )
    lines.append("")
    lines.append("### 理論的な下限（0次エントロピー、ブロック分割なし）")
    lines.append("")
    lines.append("| method | bit/sample | 理論上のratio |")
    lines.append("|---|---:|---:|")
    for row in entropy_rows:
        lines.append(f"| {row['method']} | {row['bits_per_sample']:.3f} | {row['ratio']:.3f} |")
    lines.append("")
    lines.append("## わかったこと")
    lines.append("")
    wlc_best = max(codec_rows, key=lambda r: r["ratio"])
    gp_best = max(baseline_rows, key=lambda r: r["ratio"])

    lines.append(
        f"- 提案手法（予測 + Rice符号）の中で最も良かったのは **{wlc_best['method']}**"
        f"（block_size={wlc_best['block_size']}）で、圧縮率 {wlc_best['ratio']:.2f} 倍"
        f"（{wlc_best['bits_per_sample']:.3f} bit/sample）。"
        f"全体の最良は汎用圧縮の **{gp_best['method']}**（{gp_best['ratio']:.2f} 倍）で、"
        "提案手法は汎用圧縮（特に bz2, lzma）に明確に及ばなかった。"
    )
    lines.append(
        "- `raw` 予測（差分を取らない）はほぼ効果がなく、むしろ元データより大きくなる"
        f"（ratio<1）。生波形の値は126付近を中心にばらついており、0次エントロピーだけでも"
        f"{entropy_rows[0]['bits_per_sample']:.2f} bit/sample かかる。"
        "`diff1`・`linear2` に切り替えるだけでエントロピーが"
        f"約{entropy_rows[0]['bits_per_sample'] / entropy_rows[1]['bits_per_sample']:.1f}倍改善しており、"
        "隣接サンプル間の相関が非常に強いことがわかる。"
    )
    lines.append(
        "- ブロックごとに予測手法を適応選択する `wlc/adaptive` は、固定手法よりわずかに"
        "改善するにとどまった。ブロックサイズが小さいほど適応の余地は大きくなるが、"
        "メタデータ（1ブロックあたり1バイト）や Rice パラメータの再推定コストも増えるため、"
        "小さすぎるブロックサイズではかえって不利になる。"
    )
    lines.append(
        "- 提案手法の実測値（bit/sample）は、対応する0次エントロピーの理論下限にかなり"
        "近い値まで達している。つまり **Rice符号自体の効率は妥当** であり、"
        "汎用圧縮に負けている原因は符号化方式ではなく、予測モデルが捉えている相関の"
        "射程がサンプル1〜2個分の局所的な相関に限られている点にある。"
    )
    lines.append(
        "- 波形分析（README参照）では600-1000Hz付近に強い周期成分があるとわかっている。"
        "これは65536Hzサンプリングで周期にして65〜109サンプル程度に相当し、"
        "似た波形が周期的に繰り返される。bz2やlzmaはこうした長距離の繰り返しパターンを"
        "（それぞれBWT/LZ77系のマッチングで）利用できるが、"
        "`diff1`/`linear2` のような直近1〜2点だけを見る予測器はこの周期的な繰り返しを"
        "捉えられない。これが汎用圧縮に及ばない主な理由と考えられる。"
    )
    lines.append("")
    lines.append("## 今後の改善案")
    lines.append("")
    lines.append(
        "- 周期成分を利用するため、固定オフセット（推定周期分前のサンプル）を予測に使う"
        "「長期予測（long-term prediction）」をブロックごとの候補に追加する。"
    )
    lines.append(
        "- Rice符号のパラメータ推定を、ブロック単位ではなくよりきめ細かく"
        "（あるいは適応的に）行う、またはコンテキストモデリングを導入する。"
    )
    lines.append(
        "- 可変長ブロック分割（動的計画法によるブロック境界の最適化）を検討し、"
        "予測が効きやすい区間とそうでない区間を切り分ける。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load_data()
    print(f"loaded {data.size:,} samples from {DATA_PATH}")
    print()

    print("=== proposed codec (predictor + Rice code) ===")
    codec_rows = evaluate_codec_configs(data)
    print()

    print("=== general-purpose baselines ===")
    baseline_rows = evaluate_baselines(data)
    print()

    print("=== theoretical entropy reference ===")
    entropy_rows = evaluate_entropy_reference(data)
    print()

    write_csv(codec_rows + baseline_rows + entropy_rows, RESULTS_DIR / "results.csv")
    plot_results(codec_rows, baseline_rows, entropy_rows, RESULTS_DIR / "comparison.png")

    report = render_report(codec_rows, baseline_rows, entropy_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {RESULTS_DIR / 'results.csv'}")
    print(f"wrote {RESULTS_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
