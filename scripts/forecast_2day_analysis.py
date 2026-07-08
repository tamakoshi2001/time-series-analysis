"""東京電力エリア 1時間電力需要「2日後(25〜48時間先)予測」の検証パイプライン。

レポート(report/report.tex)の数値・図をすべて生成するスクリプト。

モデル構成:
  - 基本モデル1: 364日差分で定常化した系列への ARMA
  - 基本モデル2: 状態空間モデル(ローカルレベル+日・週周期, カルマンフィルタ)
  - 応用1: ARMAX(雨・祝日の外生変数を追加)
  - 応用2: 平均パターン(年・週・日周期の平均) + 残差ARIMA(p,1,q)

検証方法(日単位ローリングオリジン):
  テスト週(直近7日)の各日 D について、需要データは「D-2日の23:00」まで手に入る
  という想定で、そこを予測起点として48時間先まで再帰的に予測する。
  1〜24時間先(=D-1日)の予測は捨て、25〜48時間先(=当日D)の24時間分だけを評価に使う。
  この手続きを起点を1日ずつずらして7回繰り返し、テスト週168時間を評価する。
  検証方式の模式図は fig_2day_scheme.png に出力する。

出力: report/assets/results_2day.json(数値) と report/assets/fig_2day_*.png(図)
実行方法: uv run python scripts/forecast_2day_analysis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 画面表示せずファイル出力のみ行うバックエンド(サーバでも動く)
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import acf, adfuller

warnings.filterwarnings("ignore")  # statsmodelsの収束警告等でログが埋まるのを防ぐ

# ---- パスと定数 ----
ROOT = Path(__file__).resolve().parent.parent   # リポジトリ直下
DATA_PATH = ROOT / "data" / "processed" / "combined_hourly_demand_labeled.csv"  # 気象・祝日ラベル付き需要データ
ASSETS = ROOT / "report" / "assets"             # 図・JSONの出力先
ASSETS.mkdir(exist_ok=True)
Y_COL = "actual_10k_kw"                         # 需要実績(万kW)の列名

TEST_DAYS = 7          # テスト週の日数(直近1週間で検証)
HORIZON_H = 48         # 予測ホライズン: 2日後 = 48時間先まで予測する
LAG_364D = 24 * 364    # 364日差分のラグ(時間単位)。364日=52週なので曜日が揃う
WINDOW_YEARS = 2       # ARMA系モデルの学習窓(年)。全期間だと計算が重く、
                       # 需要構造も変化するため直近2年に限定する

# ---- 日本語フォントの設定 ----
# 注意: plt.style.use() はrcParamsを上書きするため、スタイル適用後にフォントを設定する。
# macOSのHiragino系は.ttc(フォントコレクション)でmatplotlibが字形を解決できず
# 豆腐(□)になることがあるため、.otf/.ttf単体のフォントを優先して選ぶ。
plt.style.use("seaborn-v0_8-whitegrid")
for cand in ["YuGothic", "Hiragino Maru Gothic ProN", "Toppan Bunkyu Gothic",
             "Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans"]:
    try:
        path = font_manager.findfont(cand, fallback_to_default=False)
    except Exception:
        continue
    if path and not path.lower().endswith(".ttc"):
        plt.rcParams["font.family"] = cand
        print("Japanese font:", cand, path)
        break
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策
plt.rcParams["figure.figsize"] = (14, 5)

RESULTS: dict = {}  # レポートに載せる数値をすべてここに集めて最後にJSON保存する


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """実績と予測を時刻で突き合わせて RMSE / MAE / MAPE(%) を返す。

    RMSE: 二乗平均平方根誤差。大きい外れに敏感。
    MAE : 平均絶対誤差。誤差の平均的な大きさ。
    MAPE: 平均絶対パーセント誤差。水準によらず比較しやすい。
    """
    a = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    rmse = mean_squared_error(a["t"], a["p"]) ** 0.5
    mae = mean_absolute_error(a["t"], a["p"])
    mape = (np.abs((a["t"] - a["p"]) / a["t"])).mean() * 100
    return {"n": int(len(a)), "rmse": float(rmse), "mae": float(mae), "mape_percent": float(mape)}


# ============================================================
# データ読み込みとテスト期間の設定
# ============================================================
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
# asfreq("h"): 1時間刻みの規則的なインデックスに揃える(欠測時刻はNaNになる)。
# 時系列モデルはラグ(何時間前か)で構造を捉えるため、等間隔であることが前提。
series = df.set_index("datetime")[Y_COL].asfreq("h")
y = series.dropna()  # このデータは10年分欠損なし(87,648時間)

# 外生変数(ARMAX用)のダミー変数を作る:
#   is_rain    = 降水量が0mmより大きい時刻を1
#   is_holiday = 内閣府の祝日にあたる時刻を1
df = df.set_index("datetime")
exog_raw = pd.DataFrame({
    "is_rain": (df["precipitation_mm"].astype(float) > 0).astype(float),
    "is_holiday": df["is_holiday"].astype(float),
}).reindex(y.index).fillna(0.0)

# テスト週 = データ末尾の7日間(168時間)
test_start = y.index.max() - pd.Timedelta(hours=24 * TEST_DAYS - 1)
test = y.loc[test_start:]
test_days = sorted({ts.normalize() for ts in test.index})  # テスト対象の日付(7日分)
# 最初の予測起点: 最も早いテスト日 D に対する「D-2日の23:00」。
# モデルのパラメータ推定にはこの時点までのデータしか使わない(リーク防止)。
first_origin = test_days[0] - pd.Timedelta(days=1) - pd.Timedelta(hours=1)
train_base = y.loc[:first_origin]  # 平均パターンの推定に使う全履歴

RESULTS["data"] = {
    "rows": int(len(y)),
    "start": str(y.index.min()),
    "end": str(y.index.max()),
    "mean_demand": float(y.mean()),
    "std_demand": float(y.std()),
    "test_days": [str(d.date()) for d in test_days],
    "test_hours": int(len(test)),
    "horizon_hours": HORIZON_H,
    "first_origin": str(first_origin),
}
print("loaded:", y.index.min(), "->", y.index.max(), "n=", len(y))
print("test week:", test_days[0].date(), "->", test_days[-1].date(), " first origin:", first_origin)

# ---- 図: 検証方式の模式図 ----
# 各テスト日Dの行に「学習データ(〜D-2日23:00)」「D-1日(予測するが捨てる)」
# 「当日D(評価対象)」を色分けして描き、起点を1日ずつずらす様子を可視化する。
from matplotlib.patches import Patch  # noqa: E402

fig_base = test_days[0] - pd.Timedelta(days=3)  # 図の左端


def _xd(ts: pd.Timestamp) -> float:
    """図のx座標(左端からの経過日数)に変換する。"""
    return (ts - fig_base).total_seconds() / 86400.0


fig, ax = plt.subplots(figsize=(13, 6))
for i, d in enumerate(test_days):
    yrow = len(test_days) - 1 - i  # 上の行ほど早いテスト日
    origin = d - pd.Timedelta(days=1) - pd.Timedelta(hours=1)  # D-2日 23:00
    ax.barh(yrow, _xd(origin), left=0, height=0.6, color="#c8d6e5")            # 学習データ
    ax.barh(yrow, 1.0, left=_xd(origin) + 1 / 24, height=0.6,
            color="#feca57", hatch="//", edgecolor="white")                    # 1〜24h先(捨てる)
    ax.barh(yrow, 1.0, left=_xd(d), height=0.6, color="#ee5253", edgecolor="white")  # 25〜48h先(評価)
    ax.plot(_xd(origin), yrow + 0.42, marker="v", color="black", ms=8, clip_on=False)
ax.set_yticks(range(len(test_days)))
ax.set_yticklabels([f"テスト日 {d.strftime('%m/%d')}" for d in reversed(test_days)])
day_ticks = pd.date_range(fig_base, test_days[-1] + pd.Timedelta(days=1), freq="D")
ax.set_xticks([_xd(t) for t in day_ticks])
ax.set_xticklabels([t.strftime("%m/%d") for t in day_ticks])
ax.set_xlim(-0.05, _xd(test_days[-1] + pd.Timedelta(days=1)) + 0.05)
ax.set_xlabel("日付(各日の左端が0:00)")
ax.legend(handles=[
    Patch(color="#c8d6e5", label="学習・状態更新に使うデータ(左端よりさらに過去へ続く)"),
    Patch(facecolor="#feca57", hatch="//", edgecolor="white", label="1〜24時間先の予測(D-1日、評価しない)"),
    Patch(color="#ee5253", label="25〜48時間先の予測(当日D、評価対象)"),
    plt.Line2D([], [], marker="v", color="black", ls="none", label="予測起点(D-2日 23:00)"),
], loc="lower left", fontsize=9)
ax.set_title("検証方式: 各テスト日Dを「D-2日23:00までのデータ」から48時間先まで再帰予測(起点を1日ずつずらして7回)")
plt.tight_layout(); plt.savefig(ASSETS / "fig_2day_scheme.png", dpi=110); plt.close()


# ============================================================
# 定常性の確認 (ADF検定)
# ------------------------------------------------------------
# ARMAは弱定常な系列を前提とするため、元系列と各種差分系列にADF検定を行い、
# どの変換で定常とみなせるか・分散がどれだけ減るかを確認する。
# 364日差分 z_t = y_t - y_{t-8736} は「昨年の同じ曜日・同じ時刻」との差なので、
# 日(24h)・週(168h)・年(364d)の3つの周期を一度に取り除ける。
# ============================================================
print("== stationarity ==")
diff_full = y.diff(LAG_364D)  # 364日差分系列(先頭364日分はNaN)
transformed = {
    "level": y,             # 元系列
    "diff_1": y.diff(1),    # 1次差分(トレンド除去)
    "diff_24": y.diff(24),  # 24時間差分(日周期除去)
    "diff_168": y.diff(168),  # 168時間差分(日・週周期除去)
    "diff_364d": diff_full,   # 364日差分(日・週・年周期除去)
}
adf_rows = []
for name, x in transformed.items():
    clean = x.dropna()
    # ADF検定: 帰無仮説「単位根あり(非定常)」。p<0.05で棄却=定常とみなす。
    # autolag="AIC": 検定に含めるラグ数をAICで自動選択(最大48時間)。
    stat, p, usedlag, nobs, crit, _ = adfuller(clean, autolag="AIC", maxlag=48)
    adf_rows.append({
        "series": name, "nobs": int(nobs), "adf_stat": float(stat), "p_value": float(p),
        "used_lag": int(usedlag), "crit_5pct": float(crit["5%"]),
        "mean": float(clean.mean()), "variance": float(clean.var()),
        "stationary_5pct": bool(p < 0.05),
    })
RESULTS["adf"] = adf_rows

# ACF(自己相関)の代表値。ラグ24・168の高い相関が日・週周期の存在を示す。
acf_vals = acf(y, nlags=168, fft=True)
RESULTS["acf"] = {"lag1": float(acf_vals[1]), "lag24": float(acf_vals[24]), "lag168": float(acf_vals[168])}

# 図: 元系列と364日差分系列の推移(全期間)。
# 元系列には夏冬に高く春秋に低い年周期がはっきり見える。364日差分を取ると
# 年周期(と日・週周期)が消え、平均ほぼ0の変動だけが残ることを確認する。
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
y.plot(ax=axes[0], lw=0.3, color="#1f77b4")
axes[0].set_title("元系列: 1時間電力需要(万kW)")
axes[0].set_ylabel("需要(万kW)")
diff_full.plot(ax=axes[1], lw=0.3, color="#2ca02c")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("364日差分系列(年・週・日周期を除去)")
axes[1].set_ylabel("差分(万kW)"); axes[1].set_xlabel("日時")
plt.tight_layout(); plt.savefig(ASSETS / "fig_2day_series.png", dpi=110); plt.close()

# 図: 元系列と差分系列のACF(2週間分のラグまで)。
# 赤破線=ラグ24(日周期)、紫破線=ラグ168(週周期)の位置。
fig, axes = plt.subplots(4, 1, figsize=(14, 14))
for ax, (title, x) in zip(axes, [
    ("元系列のACF", y),
    ("24時間差分後のACF", y.diff(24)),
    ("168時間差分後のACF", y.diff(168)),
    ("364日差分後のACF", diff_full),
]):
    plot_acf(x.dropna(), lags=24 * 14, ax=ax, alpha=0.05)
    ax.set_title(title); ax.axvline(24, color="red", ls="--"); ax.axvline(168, color="purple", ls="--")
plt.tight_layout(); plt.savefig(ASSETS / "fig_2day_acf.png", dpi=110); plt.close()
print("stationarity done")


# ============================================================
# 基本モデル1: 364日差分 + ARMA / 応用1: ARMAX (雨・祝日)
# ------------------------------------------------------------
# 差分系列 z_t にARMA(p,q)を当て、予測は ŷ = y_{t-8736} + ẑ で水準に戻す。
# ARMAXは z_t の式に外生変数の回帰項 β'x_t を加えたもの。
# ============================================================
print("== ARMA / ARMAX on 364-day diff ==")
win_start = first_origin - pd.Timedelta(days=365 * WINDOW_YEARS)  # 学習窓の開始(直近2年)
fit_diff = diff_full.loc[win_start:first_origin].dropna()         # パラメータ推定用の差分系列

# 外生変数も364日差分にする。目的変数が z_t = y_t - y_{t-8736} なので、
# 外生変数も同じ変換 Δx_t = x_t - x_{t-8736} を施すのが整合的
# (例: Δ祝日=+1 は「今年は祝日だが前年同時刻は平日」を意味する)。
exog_diff = (exog_raw - exog_raw.shift(LAG_364D)).dropna()
exog_diff.columns = ["d_is_rain", "d_is_holiday"]
fit_exog = exog_diff.reindex(fit_diff.index)

# 候補次数 (p,d,q)。d=0(差分済みの系列に当てるため)。AICで選択する。
arma_orders = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2)]


def fit_candidates(prefix: str, exog: pd.DataFrame | None):
    """候補次数のARMA(X)を全てフィットし、AIC昇順の結果リストとフィット辞書を返す。

    AIC = -2*対数尤度 + 2*パラメータ数。当てはまりと複雑さのトレードオフを取り、
    小さいほど良い。レポートには参考として係数とp値も載せるため一緒に記録する。
    enforce_stationarity/invertibility=False は、推定を制約なしで行い
    最適化の失敗を避けるための設定。
    """
    rows, fits = [], {}
    for order in arma_orders:
        label = f"{prefix}{(order[0], order[2])}"  # 例: ARMA(2, 2)
        try:
            res = ARIMA(fit_diff, exog=exog, order=order,
                        enforce_stationarity=False, enforce_invertibility=False).fit()
            fits[label] = (order, res)
            rows.append({
                "model": label, "order": list(order),
                "aic": float(res.aic), "bic": float(res.bic),
                "params": {k: float(v) for k, v in res.params.items()},
                "pvalues": {k: float(v) for k, v in res.pvalues.items()},
            })
            print("  fitted", label, "aic=", round(res.aic, 1))
        except Exception as e:  # noqa: BLE001 - 収束失敗した次数は表に理由だけ残す
            rows.append({"model": label, "order": list(order), "error": str(e)})
    rows = sorted(rows, key=lambda r: r.get("aic", 1e18))  # AIC昇順(先頭が最良)
    return rows, fits


# ARMA: 外生変数なし / ARMAX: 雨・祝日ダミー付き。それぞれAICで次数選択。
RESULTS["arma_order_selection"], fitted = fit_candidates("ARMA", None)
best_arma = RESULTS["arma_order_selection"][0]["model"]
best_order, arma_res = fitted[best_arma]
RESULTS["arma_best"] = best_arma
print("best ARMA:", best_arma)

RESULTS["armax_order_selection"], armax_fitted = fit_candidates("ARMAX", fit_exog)
best_armax = RESULTS["armax_order_selection"][0]["model"]
armax_best_order, armax_res = armax_fitted[best_armax]
RESULTS["armax_best"] = best_armax
print("best ARMAX:", best_armax)

# --- ローリング2日後予測 ---
ref_364d = y.shift(LAG_364D)  # 前年値 y_{t-8736}。差分予測を水準に戻すときに足す


def rolling_2day(res, use_exog: bool) -> pd.Series:
    """各テスト日Dを「D-2日23:00」起点で48時間先まで再帰予測し、当日D分をつなげて返す。

    1〜24ステップ目(=D-1日)の予測は評価に使わず、25〜48ステップ目(=当日D)だけを
    採用する。起点を1日ずつずらして7回繰り返す(日単位ローリング)。

    ポイントは res.apply(endog):
      - パラメータは再推定せず(first_origin時点の推定値を固定)、
        起点までのデータでカルマンフィルタだけを回し直して予測の初期状態を更新する。
      - これにより「パラメータは過去に固定、直近データは予測に反映」という
        実運用に近いローリング予測を高速に実現できる。
    ARMAXの場合は、予測期間の外生変数(祝日は既知、雨は実績で代用)も渡す。
    """
    preds = []
    for d in test_days:
        origin = d - pd.Timedelta(days=1) - pd.Timedelta(hours=1)   # D-2日 23:00
        endog = diff_full.loc[win_start:origin].dropna()            # 起点までの差分系列
        fut_idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON_H, freq="h")
        if use_exog:
            applied = res.apply(endog, exog=exog_diff.reindex(endog.index))
            dpred = applied.forecast(steps=HORIZON_H, exog=exog_diff.reindex(fut_idx))
        else:
            applied = res.apply(endog)
            dpred = applied.forecast(steps=HORIZON_H)
        dpred.index = fut_idx
        day_idx = fut_idx[24:]  # 25〜48時間先 = 当日D の24時間
        # 差分の予測 ẑ を前年値に足して水準へ: ŷ = y_{t-8736} + ẑ
        preds.append(ref_364d.reindex(day_idx) + dpred.iloc[24:])
    return pd.concat(preds)


fc = {}  # モデル名 -> テスト週168時間の予測系列
# 全候補次数について2日後ローリング精度を計算し、次数選択表に参考として付記する
# (AIC最良の次数だけを総合比較 fc に採用する)。
for row in RESULTS["arma_order_selection"]:
    order, res = fitted[row["model"]]
    pred = rolling_2day(res, use_exog=False)
    row.update({f"fc_{k}": v for k, v in metrics(test, pred).items()})
    if row["model"] == best_arma:
        fc["ARMA_diff364d"] = pred
print("ARMA rolling done")
for row in RESULTS["armax_order_selection"]:
    order, res = armax_fitted[row["model"]]
    pred = rolling_2day(res, use_exog=True)
    row.update({f"fc_{k}": v for k, v in metrics(test, pred).items()})
    if row["model"] == best_armax:
        fc["ARMAX_rain_holiday"] = pred
print("ARMAX rolling done")

# ============================================================
# ベースライン
# ============================================================
fc["naive_2day"] = y.shift(48).reindex(test.index)  # 「2日前と同じ」という素朴な予測
# 前年値コピーは「ARMA≈前年値」の実証用の参考値(総合比較のランキングには含めない)
naive_lastyear = ref_364d.reindex(test.index)
RESULTS["naive_lastyear_reference"] = metrics(test, naive_lastyear)


# ============================================================
# 検証:「ARMA/ARMAXの2日後予測 ≈ 前年値コピー」の定量比較
# ------------------------------------------------------------
# 定常なARMAのhステップ先予測はhが大きくなると系列の平均に減衰するため、
# 48時間先では ẑ ≈ (定数) となり、ŷ ≈ y_{t-8736} + 定数 になるはず。
# それを「前年値との平均絶対差・相関」で確かめる。
# ============================================================
mean_demand = float(test.mean())
closeness = {}
for label in ["ARMA_diff364d", "ARMAX_rain_holiday"]:
    d = (fc[label] - naive_lastyear).dropna()  # 予測と前年値の差 = ARMAによる補正量
    closeness[label] = {
        "mean_abs_diff_10k_kw": float(d.abs().mean()),
        "max_abs_diff_10k_kw": float(d.abs().max()),
        "mean_abs_diff_pct_of_demand": float(d.abs().mean() / mean_demand * 100),
        "corr_with_lastyear": float(fc[label].corr(naive_lastyear)),
    }
RESULTS["closeness_to_lastyear"] = closeness
print("closeness:", json.dumps(closeness, indent=1))

# 図: 実績・前年値・ARMA/ARMAX予測の重ね描き(ARMA系が前年値に重なることを見せる)
fig, ax = plt.subplots(figsize=(14, 6))
test.plot(ax=ax, lw=2.2, color="black", label="実績")
naive_lastyear.plot(ax=ax, lw=1.6, color="#1f77b4", label="前年値 y(t-364日)")
fc["ARMA_diff364d"].plot(ax=ax, lw=1.2, ls="--", color="#d62728", label="364日差分+ARMA")
fc["ARMAX_rain_holiday"].plot(ax=ax, lw=1.2, ls=":", color="#2ca02c", label="364日差分+ARMAX")
ax.set_title("2日後予測: 364日差分+ARMA/ARMAXは前年値とほぼ重なる")
ax.set_xlabel("日時"); ax.set_ylabel("需要(万kW)"); ax.legend()
plt.tight_layout(); plt.savefig(ASSETS / "fig_2day_arma_vs_lastyear.png", dpi=110); plt.close()


# ============================================================
# 応用2: 平均パターン(年・週・日周期の平均) + 残差ARIMA
# ------------------------------------------------------------
# 周期成分を「前年の1観測」ではなく「10年分の平均」で表す。
#   ŷ_t = m + d(時刻) + w(曜日,時刻) + a(通日) + (残差のARIMA予測)
# 平均化により単年の天候ノイズは約 1/sqrt(10) に圧縮される。
# 成分は「日→週→年」の順に逐次推定する(先に引いた成分の影響を
# 残りから除いてから次を推定することで、二重計上を防ぐ)。
# ============================================================
print("== seasonal profile ==")
tr = train_base            # 平均パターンは first_origin までの全履歴(約10年)から推定
m = float(tr.mean())       # (1) 全体平均
# (2) 日周期: 時刻(0..23)ごとの平均の、全体平均からの偏差(24値)
daily = tr.groupby(tr.index.hour).mean() - m
after1 = tr - pd.Series(pd.Index(tr.index.hour).map(daily), index=tr.index)  # 日周期を除去
# (3) 週周期: 日周期除去後の系列で、(曜日,時刻)ごとの平均の偏差(7x24=168値)
weekly = after1.groupby([after1.index.dayofweek, after1.index.hour]).mean() - after1.mean()
after2 = after1 - pd.Series(
    weekly.reindex(pd.MultiIndex.from_arrays([tr.index.dayofweek, tr.index.hour])).to_numpy(),
    index=tr.index).fillna(0)  # 週周期も除去
# (4) 年周期: 通日(1..366)ごとの平均の偏差。日単位のギザギザを均すため
#     31日中心移動平均で平滑化する(年周期は滑らかに変わるはずという事前知識)。
yr_raw = after2.groupby(after2.index.dayofyear).mean() - after2.mean()
yearly = yr_raw.rolling(31, center=True, min_periods=7).mean().bfill().ffill()


def seasonal_profile(index: pd.DatetimeIndex) -> pd.Series:
    """任意の時刻indexに対する平均パターン(全体平均+日+週+年周期)を返す。

    パターンは「時刻」「曜日」「通日」だけで決まるので、未来の時刻に対しても
    そのまま計算できる(=予測に使える)。
    """
    d = pd.Series(pd.Index(index.hour).map(daily), index=index)
    w = pd.Series(weekly.reindex(pd.MultiIndex.from_arrays([index.dayofweek, index.hour])).to_numpy(),
                  index=index).fillna(0)
    a = pd.Series(pd.Index(index.dayofyear).map(yearly), index=index).fillna(0)
    return m + d + w + a


# --- 残差にARIMA(p,1,q)を載せる ---
# 平均パターンを引いた残差には、気温など天候起因の「持続的な水準ずれ」が残る
# (ACFの減衰が極めて遅い)。定常ARMA(d=0)だと予測が数時間で平均0に減衰して
# 直近のずれを活かせないため、1次差分(d=1)を入れて水準ずれを先へ引き継ぐ。
des_full = y - seasonal_profile(y.index)                 # 残差系列(全期間)
des_fit = des_full.loc[win_start:first_origin].dropna()  # 推定用(直近2年)
des_acf = acf(des_fit, nlags=168, fft=True)              # 残差の持続性の根拠として記録
RESULTS["profile_residual_acf"] = {
    "lag1": float(des_acf[1]), "lag24": float(des_acf[24]), "lag168": float(des_acf[168]),
}

# 残差モデルの次数候補(すべてd=1)。AICで選択。
des_orders = [(1, 1, 1), (2, 1, 1), (2, 1, 2)]
des_rows, des_fitted = [], {}
for order in des_orders:
    label = f"ARIMA{order}"
    res = ARIMA(des_fit, order=order,
                enforce_stationarity=False, enforce_invertibility=False).fit()
    des_fitted[label] = (order, res)
    des_rows.append({"model": label, "order": list(order),
                     "aic": float(res.aic), "bic": float(res.bic)})
    print("  fitted residual", label, "aic=", round(res.aic, 1))
RESULTS["profile_arima_orders"] = sorted(des_rows, key=lambda r: r["aic"])
des_best = RESULTS["profile_arima_orders"][0]["model"]
des_best_order, des_res = des_fitted[des_best]
RESULTS["profile_arima_best"] = des_best
print("best residual ARIMA:", des_best)

# 日単位ローリング2日後予測(仕組みはrolling_2dayと同じ。系列が残差、足し戻す相手が平均パターン)
preds = []
for d in test_days:
    origin = d - pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    endog = des_full.loc[win_start:origin].dropna()
    applied = des_res.apply(endog)  # パラメータ固定のまま起点までの残差で状態を更新
    fut_idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON_H, freq="h")
    rp = applied.forecast(steps=HORIZON_H)
    rp.index = fut_idx
    day_idx = fut_idx[24:]  # 25〜48時間先 = 当日D
    preds.append(seasonal_profile(day_idx) + rp.iloc[24:])  # ŷ = 平均パターン + 残差予測
fc["seasonal_profile_plus_ARIMA"] = pd.concat(preds)
print("seasonal profile done")

# 図: 平均パターンの中身(日・週・年の3成分)
fig, axes = plt.subplots(3, 1, figsize=(14, 11))
daily.plot(ax=axes[0], marker="o", title="日周期パターン(時刻別平均の偏差, 万kW)")
axes[0].set_xlabel("時刻")
weekly.reset_index(drop=True).plot(ax=axes[1], lw=1.2, title="週周期パターン(曜日×時刻の偏差, 万kW)")
axes[1].set_xticks(range(0, 169, 24))
axes[1].set_xticklabels(["月", "火", "水", "木", "金", "土", "日", ""])
yearly.plot(ax=axes[2], lw=1.5, title="年周期パターン(通日別平均・31日移動平均, 万kW)")
axes[2].set_xlabel("通日(1〜366)")
plt.tight_layout(); plt.savefig(ASSETS / "fig_2day_profile.png", dpi=110); plt.close()


# ============================================================
# 基本モデル2: 状態空間モデル(カルマンフィルタ)
# ------------------------------------------------------------
# 観測方程式:   y_t = mu_t + gamma24_t + gamma168_t + eps_t
# システム方程式: mu_{t+1} = mu_t + eta_t (ランダムウォークのレベル)
#               gamma: 三角関数型の確率的季節成分(回転行列で時間発展)
#
# statsmodelsのUnobservedComponentsを使用:
#   level="llevel"  -> ローカルレベル(ランダムウォーク)
#   freq_seasonal   -> 三角関数型季節成分。日周期(24h)は8調波、週周期(168h)は
#                      6調波に制限して状態次元と計算量を抑える。
# 分散パラメータ(観測ノイズ・レベル・各季節成分)は最尤法で推定し、
# 状態はカルマンフィルタで逐次推定する。
# レベル mu_t が直近の水準ずれに自動的に追随する点が、残差ARIMAのd=1と同じ役割。
# ============================================================
print("== state space (Kalman filter) ==")
uc_model = UnobservedComponents(
    y.loc[win_start:first_origin], level="llevel",
    freq_seasonal=[{"period": 24, "harmonics": 8}, {"period": 168, "harmonics": 6}],
)
uc_res = uc_model.fit(disp=False, maxiter=200)  # 最尤推定(数値最適化)
RESULTS["state_space"] = {
    "spec": "local level + freq_seasonal(period=24, 8 harmonics) + freq_seasonal(period=168, 6 harmonics)",
    "aic": float(uc_res.aic),
    "params": {k: float(v) for k, v in uc_res.params.items()},
}
print("state space fitted. aic=", round(uc_res.aic, 1))

# 日単位ローリング2日後予測。res.apply()で分散パラメータは固定したまま、
# 起点(D-2日23:00)までの観測でカルマンフィルタを回して状態(レベル・季節成分)を更新
# →48時間先まで予測し当日D分を採用。水準そのものを予測しているので足し戻し不要。
preds = []
for d in test_days:
    origin = d - pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    applied = uc_res.apply(y.loc[win_start:origin])
    fut_idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON_H, freq="h")
    rp = applied.forecast(steps=HORIZON_H)
    rp.index = fut_idx
    preds.append(rp.iloc[24:])
fc["state_space_kalman"] = pd.concat(preds)
print("state space rolling done")


# ============================================================
# 総合比較: 全モデルをRMSEで順位付けし、テスト週の予測を重ね描き
# ============================================================
print("== ranking ==")
rank = [{"model": k, **metrics(test, v)} for k, v in fc.items()]
RESULTS["forecast_ranking"] = sorted(rank, key=lambda r: r["rmse"])
for r in RESULTS["forecast_ranking"]:
    print(f"  {r['model']:32s} rmse={r['rmse']:7.1f} mape={r['mape_percent']:.2f}%")

fig, ax = plt.subplots(figsize=(14, 6))
test.plot(ax=ax, lw=2.4, color="black", label="実績")
for label, color, ls in [
    ("ARMA_diff364d", "#d62728", "--"),
    ("ARMAX_rain_holiday", "#2ca02c", ":"),
    ("seasonal_profile_plus_ARIMA", "#ff7f0e", "--"),
    ("state_space_kalman", "#9467bd", "-."),
]:
    fc[label].plot(ax=ax, lw=1.2, ls=ls, color=color, label=label)
ax.set_title("テスト週の実績と2日後予測(各日を2日前までのデータから予測)")
ax.set_xlabel("日時"); ax.set_ylabel("需要(万kW)"); ax.legend()
plt.tight_layout(); plt.savefig(ASSETS / "fig_2day_forecast_compare.png", dpi=110); plt.close()

# ============================================================
# 保存: レポートに載せる数値をまとめてJSONに書き出す
# ============================================================
out = ASSETS / "results_2day.json"
with open(out, "w") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)
print("SAVED", out)
print("best:", RESULTS["forecast_ranking"][0])
