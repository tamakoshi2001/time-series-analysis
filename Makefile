# 使い方:
#   make analysis  分析パイプラインを実行(図とJSONを report/assets/ に再生成)
#   make report    レポートPDFをコンパイル(report/report.pdf)
#   make clean     LaTeXの中間ファイルを削除

.PHONY: analysis report clean

analysis:
	uv run python scripts/forecast_2day_analysis.py

report:
	latexmk -cd -lualatex -interaction=nonstopmode report/report.tex

clean:
	latexmk -cd -C report/report.tex
