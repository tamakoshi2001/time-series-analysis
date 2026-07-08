"""時別需要データに東京の気象と日本の祝日のラベルを付与する。

data/processed/combined_hourly_demand.csv (2016-04〜の時別需要) に対して、
  - Open-Meteo アーカイブAPIから東京の時別気象(気温・湿度・降水・天候コード等)
  - 内閣府の祝日CSV
を取得してマージし、曜日・祝日・day_type などの列を加えた
data/processed/combined_hourly_demand_labeled.csv を書き出す(分析パイプラインの入力)。

取得した生データは data/raw/weather/ と data/raw/holiday/ にも保存する。
実行方法: uv run python scripts/add_tokyo_weather_holiday_labels.py (要ネットワーク)
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


DEMAND_PATH = Path("data/processed/combined_hourly_demand.csv")
OUTPUT_PATH = Path("data/processed/combined_hourly_demand_labeled.csv")
WEATHER_DIR = Path("data/raw/weather")
HOLIDAY_DIR = Path("data/raw/holiday")
WEATHER_PATH = WEATHER_DIR / "tokyo_hourly_weather_open_meteo.csv"
HOLIDAY_PATH = HOLIDAY_DIR / "japan_holidays_cao.csv"

TOKYO_LATITUDE = 35.6895
TOKYO_LONGITUDE = 139.6917
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CAO_HOLIDAY_CSV_URL = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"


WEATHER_CODE_LABELS = {
    0: "clear",
    1: "mainly_clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing_rime_fog",
    51: "light_drizzle",
    53: "moderate_drizzle",
    55: "dense_drizzle",
    56: "light_freezing_drizzle",
    57: "dense_freezing_drizzle",
    61: "slight_rain",
    63: "moderate_rain",
    65: "heavy_rain",
    66: "light_freezing_rain",
    67: "heavy_freezing_rain",
    71: "slight_snow",
    73: "moderate_snow",
    75: "heavy_snow",
    77: "snow_grains",
    80: "slight_rain_showers",
    81: "moderate_rain_showers",
    82: "violent_rain_showers",
    85: "slight_snow_showers",
    86: "heavy_snow_showers",
    95: "thunderstorm",
    96: "thunderstorm_with_slight_hail",
    99: "thunderstorm_with_heavy_hail",
}


def fetch_json(url: str, params: dict[str, str]) -> dict:
    full_url = f"{url}?{urlencode(params)}"
    with urlopen(full_url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def yearly_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[date, date]]:
    """期間を暦年単位に分割する(Open-Meteo APIへの1リクエストを1年分に抑える)。"""
    ranges: list[tuple[date, date]] = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start.date(), date(year, 1, 1))
        chunk_end = min(end.date(), date(year, 12, 31))
        ranges.append((chunk_start, chunk_end))
    return ranges


def fetch_tokyo_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Open-Meteoアーカイブから東京の時別気象を取得してDataFrameで返す。"""
    frames = []
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
        "shortwave_radiation",
    ]

    for start_date, end_date in yearly_ranges(start, end):
        payload = fetch_json(
            OPEN_METEO_ARCHIVE_URL,
            {
                "latitude": str(TOKYO_LATITUDE),
                "longitude": str(TOKYO_LONGITUDE),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(hourly_vars),
                "timezone": "Asia/Tokyo",
            },
        )
        hourly = payload["hourly"]
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(hourly["time"]),
                "temperature_2m_c": hourly["temperature_2m"],
                "relative_humidity_2m_pct": hourly["relative_humidity_2m"],
                "precipitation_mm": hourly["precipitation"],
                "weather_code": hourly["weather_code"],
                "wind_speed_10m_kmh": hourly["wind_speed_10m"],
                "shortwave_radiation_w_m2": hourly["shortwave_radiation"],
            }
        )
        frames.append(frame)

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    weather["weather_code"] = pd.to_numeric(weather["weather_code"], errors="coerce").astype("Int64")
    weather["weather_label"] = weather["weather_code"].map(WEATHER_CODE_LABELS).fillna("unknown")
    return weather


def fetch_japan_holidays() -> pd.DataFrame:
    """内閣府の祝日CSVを取得・保存し、日付と祝日名のDataFrameで返す。"""
    raw = fetch_bytes(CAO_HOLIDAY_CSV_URL)
    HOLIDAY_PATH.write_bytes(raw)  # 再現性のため生データも保存しておく

    text = raw.decode("cp932")  # 内閣府のCSVはShift_JIS系(cp932)
    rows = list(csv.reader(text.splitlines()))
    header = rows[0]
    holiday_date_col = header[0]
    holiday_name_col = header[1]
    holidays = pd.DataFrame(rows[1:], columns=header)
    holidays = holidays.rename(
        columns={
            holiday_date_col: "calendar_date",
            holiday_name_col: "holiday_name",
        }
    )
    holidays["calendar_date"] = pd.to_datetime(holidays["calendar_date"], format="mixed").dt.date
    holidays = holidays.drop_duplicates(subset=["calendar_date"]).sort_values("calendar_date")
    return holidays


def main() -> None:
    WEATHER_DIR.mkdir(exist_ok=True)
    HOLIDAY_DIR.mkdir(exist_ok=True)

    # 需要データを読み、気象データを取得すべき期間(需要データの範囲)を決める
    demand = pd.read_csv(DEMAND_PATH, parse_dates=["datetime"])
    demand = demand.sort_values("datetime").reset_index(drop=True)
    start = demand["datetime"].min()
    end = demand["datetime"].max()

    # 東京の時別気象を取得し、生データとして保存
    weather = fetch_tokyo_weather(start, end)
    weather.to_csv(WEATHER_PATH, index=False)

    holidays = fetch_japan_holidays()

    # 気象はdatetime(時刻)で、祝日はcalendar_date(日付)で需要にマージする
    labeled = demand.merge(weather, on="datetime", how="left")
    labeled["calendar_date"] = labeled["datetime"].dt.date
    labeled["year"] = labeled["datetime"].dt.year
    labeled["month"] = labeled["datetime"].dt.month
    labeled["day"] = labeled["datetime"].dt.day
    labeled["hour"] = labeled["datetime"].dt.hour
    labeled["weekday"] = labeled["datetime"].dt.dayofweek
    labeled["weekday_name"] = labeled["datetime"].dt.day_name()
    labeled["is_weekend"] = labeled["weekday"].isin([5, 6])

    labeled = labeled.merge(holidays, on="calendar_date", how="left")
    # 祝日名がマージできた行 = 祝日。分析ではこのis_holidayをダミー変数に使う
    labeled["is_holiday"] = labeled["holiday_name"].notna()
    labeled["holiday_name"] = labeled["holiday_name"].fillna("")
    labeled["is_weekend_or_holiday"] = labeled["is_weekend"] | labeled["is_holiday"]
    # day_typeは weekday < weekend < holiday の優先順で上書き(祝日が最優先)
    labeled["day_type"] = "weekday"
    labeled.loc[labeled["is_weekend"], "day_type"] = "weekend"
    labeled.loc[labeled["is_holiday"], "day_type"] = "holiday"

    labeled.to_csv(OUTPUT_PATH, index=False)

    missing_weather = labeled["temperature_2m_c"].isna().sum()
    print(f"Wrote {OUTPUT_PATH} rows={len(labeled)}")
    print(f"Weather rows={len(weather)} saved={WEATHER_PATH}")
    print(f"Holiday rows={len(holidays)} saved={HOLIDAY_PATH}")
    print(f"Range={start} to {end}")
    print(f"Missing weather rows={missing_weather}")
    print(f"Holiday-labeled hourly rows={int(labeled['is_holiday'].sum())}")


if __name__ == "__main__":
    main()
