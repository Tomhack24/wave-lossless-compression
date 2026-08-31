# Wave Lossless Compression

このリポジトリは、8bit の波形データを対象に、可逆圧縮の方法を調べるためのものです。

入力データは `data/wave_2026.dat` です。`uint8` の生データとして読み込みます。
`docs/document.pdf` の記述をもとに、サンプリング周波数は `65536 Hz`、サンプル数は
`3932160`、長さは `60` 秒として扱っています。

## 目的

- 波形や周波数成分を可視化する。
- 圧縮しやすい可逆変換を調べる。
- ブロックごとに予測手法を選ぶ可逆圧縮器の土台を作る。

## ディレクトリ構成

```text
data/
  wave_2026.dat                 # uint8 の生波形データ

docs/
  document.pdf                  # データに関する参考資料

scripts/
  analyze_raw_data.py           # 波形、ヒストグラム、FFT、スペクトログラムを出力するスクリプト
  evaluate_compression.py       # 複数の圧縮設定・汎用圧縮と比較し、report/ にレポートを出力するスクリプト

src/wave_lossless_compression/
  predictors.py                 # raw / diff1 / linear2 の可逆な予測残差変換
  rice.py                       # Rice符号（ゼロ次エントロピーに基づくkの自動選択を含む）
  codec.py                      # ブロック単位で予測手法を選択し、Rice符号で実際に圧縮/復元するコーデック

tests/                          # predictors / rice / codec の可逆性・境界条件のテスト

results/raw_data_analysis/
  overview.png                  # 分析結果の概要
  waveform.png                  # 波形の包絡表示
  histogram.png                 # 値の分布
  fft_spectrum.png              # 全体に対するFFT
  spectrogram_0_2khz.png        # 0-2kHz に絞ったスペクトログラム
  spectrogram_full.png          # 全帯域のスペクトログラム
  summary.txt                   # 基本統計量

results/compression_evaluation/
  results.csv                   # 各設定の圧縮サイズ・圧縮率・処理時間
  comparison.png                # 手法ごとの圧縮率を比較したグラフ

report/
  compression_report.md         # 圧縮手法の比較レポート（本文）
```

## データ分析

以下を実行すると、`data/wave_2026.dat` を読み込み、分析結果を
`results/raw_data_analysis/` に保存します。

```bash
./.venv/bin/python scripts/analyze_raw_data.py
```

現在の分析では、以下のような特徴が見えています。

- 値は `126` 付近に強く集中している。
- データ長は `65536 Hz` で 60 秒分。
- `600-1000 Hz` 付近に強い周波数成分がある。
- スペクトログラムを見ると、この帯域は時間全体にわたって存在しているが、強度は時間によって変化している。

## 予測残差

`src/wave_lossless_compression/predictors.py` には、ブロック単位で使うための可逆変換を実装しています。

- `raw_encode` / `raw_decode`: 値をそのまま扱う。
- `diff1_encode` / `diff1_decode`: 1つ前の値との差分を扱う。
- `linear2_encode` / `linear2_decode`: 直前2点から次の値を線形予測し、その残差を扱う。

どの変換も可逆です。

```text
decode(encode(block)) == block
```

これらの残差は、`src/wave_lossless_compression/rice.py` の Rice 符号で実際にビット列へ符号化しています。

## Rice 符号とコーデック

- `rice.py`: 符号なし整数列を Rice 符号でビット列に変換します。ブロックごとに、符号長が最小になる
  パラメータ `k` を総当たりで自動選択します（`best_k`）。負の残差は zigzag 変換で符号なし整数に
  変換してから符号化します。
- `codec.py`: `compress` / `decompress` が実際の圧縮・復元を行うコーデックです。

```text
uint8 の生データ
-> 固定長ブロックに分割
-> ブロックごとに raw / diff1 / linear2 を試し、Rice符号で最も小さくなる予測手法とkを選ぶ
-> 選んだ予測手法(2bit) + k(6bit) をメタデータとして1バイト保存
-> 残差をzigzag変換してRice符号化し、ビット列として連結する
```

`compress(data, block_size, predictors)` で圧縮し、`decompress(payload)` で完全に元の
`uint8` 配列を復元できます（可逆）。`predictors` に渡す組を1つに絞れば固定手法、複数渡せば
ブロックごとの適応選択になります。

## 圧縮結果の評価

```bash
./.venv/bin/python scripts/evaluate_compression.py
```

を実行すると、いくつかのブロックサイズ・予測手法の組み合わせで実際に圧縮・復元を行い、
可逆性を確認したうえで、`zlib` / `bz2` / `lzma` の汎用圧縮とも比較します。結果は
`results/compression_evaluation/`（CSVとグラフ）と `report/compression_report.md`
（まとめレポート）に出力されます。

現時点でわかっていることの要約（詳細は `report/compression_report.md` を参照）:

- `raw`（差分なし）はほぼ効果がなく、`diff1` / `linear2` に切り替えるだけでエントロピーが
  大きく下がる。隣接サンプル間の相関が強いデータであるため。
- 提案手法（予測 + Rice符号、ブロックサイズ16384、適応選択）で約 **6.0倍** の圧縮率。
- ただし汎用圧縮の `bz2` は約 **17.0倍**、`lzma` は約 **13.2倍** に達し、提案手法を大きく
  上回る。波形に含まれる周期成分（600-1000Hz）のような長距離の繰り返しを、直近1〜2点しか
  見ない予測器では捉えられないことが主な原因と考えられる。

## 今後の圧縮方針

- 周期成分を利用する長期予測（long-term prediction）をブロックごとの候補に追加する。
- 可変長ブロック分割（動的計画法によるブロック境界の最適化）を検討する。
- Rice符号より高度なエントロピー符号化（コンテキストモデリングなど）を検討する。
