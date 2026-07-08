# 東京電力エリア 1時間電力需要の2日後予測

東京電力エリアの1時間電力需要（2016-04〜2026-03、10年分）を対象に、
2日後（25〜48時間先）の需要予測モデルを構築・検証するプロジェクト。
成果物はレポート `report/report.pdf`。

## 使い方

```bash
uv sync            # 依存関係のセットアップ
make analysis      # 分析パイプラインを実行(図とJSONを report/assets/ に再生成)
make report        # レポートPDFをコンパイル(report/report.pdf, 要TeX Live)
```

## ディレクトリ構成

```
data/
  raw/
    juyo/          東京電力の需要実績CSV(juyo-2016〜2022)
    zip/           でんき予報の月別ZIP(2022-04〜)
    daily/         ZIPを展開した日別CSV
    weather/       Open-Meteoの東京時別気象データ
    holiday/       内閣府の祝日CSV
  processed/
    combined_hourly_demand_labeled.csv  気象・祝日ラベル付き(分析の入力)
scripts/           データセット生成・分析スクリプト
report/            レポートのLaTeXソースとPDF
  assets/          分析の出力(図・results_2day.json・実行ログ)
docs/              試問対策メモなど
```

## データ

分析の入力は `data/processed/combined_hourly_demand_labeled.csv`。
data/raw/ の生データ(juyo実績 2016-04〜2022-03 + でんき予報日別CSV 2022-04〜)を
時別に結合し、東京の気象(Open-Meteo)と祝日(内閣府)のラベルを付与したもの。

生データから一括再生成できる:

```bash
uv run python scripts/build_dataset.py
```

気象・祝日は data/raw/ に保存済みの生データがあればそれを使い(オフライン可)、
無ければAPIから取得して保存する。

※ `data/` は再配布権の観点からリポジトリに含めていない。
生データの入手元は東京電力パワーグリッド「でんき予報」の過去実績ダウンロード
(https://www.tepco.co.jp/forecast/html/download-j.html)、
Open-Meteo、内閣府の祝日CSV(詳細はレポートの出典を参照)。

## 分析

`scripts/forecast_2day_analysis.py` が分析本体。
364日差分+ARMA/ARMAX、平均パターン+残差ARIMA、
状態空間モデル(カルマンフィルタ)をローリングオリジン方式で検証する。
