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

src/wave_lossless_compression/
  predictors.py                 # raw / diff1 / linear2 の可逆な予測残差変換

results/raw_data_analysis/
  overview.png                  # 分析結果の概要
  waveform.png                  # 波形の包絡表示
  histogram.png                 # 値の分布
  fft_spectrum.png              # 全体に対するFFT
  spectrogram_0_2khz.png        # 0-2kHz に絞ったスペクトログラム
  spectrogram_full.png          # 全帯域のスペクトログラム
  summary.txt                   # 基本統計量
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

これらの残差を、後段で Rice 符号や Golomb 符号などに渡して圧縮する想定です。

## 今後の圧縮方針

想定している圧縮の流れは以下です。

```text
uint8 の生データ
-> ブロックに分割
-> ブロックごとに複数の予測手法を試す
-> 最もコストの小さい残差表現を選ぶ
-> 残差をエントロピー符号化する
-> 復元に必要なメタデータを保存する
```

まずは固定長ブロックで、`raw`、`diff1`、`linear2` のどれがよいかをブロックごとに選ぶ方針です。
その後、必要であれば動的計画法による可変長ブロック分割も検討します。
