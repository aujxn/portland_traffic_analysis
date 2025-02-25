import polars as pl
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import time
from traffic_data_analysis.utils import load_directional_hourly

df_traffic, df_meta  = load_directional_hourly()
keep_years = [2018, 2019, 2023, 2024]

df = (
    df_traffic
    .filter(pl.col("Year").is_in(keep_years))
    .filter(pl.col("Weekend") == False)
)

df_grouped = (
    df.group_by(["LocationID", "Direction", "Year", "Hour"])
    .agg(pl.col("Volume").median().alias("median"))
)

hour_knots = np.arange(25)
hour_fine = np.linspace(0, 24, 500)

results = []
unique_combinations = df_grouped.select(["LocationID", "Direction", "Year"]).unique()
for row in unique_combinations.iter_rows(named=True):
    location_id = row["LocationID"]
    direction = row["Direction"]
    year = row["Year"]

    group = (
        df_grouped
        .filter(
            (pl.col("LocationID") == location_id) &
            (pl.col("Direction") == direction) &
            (pl.col("Year") == year)
        )
        .sort("Hour")
    )

    y = group["median"].to_list()
    if len(y) != 24:
        print(f"Missing hours for {location_id} {direction} Year={year}. Found {len(y)} hours.")
        continue

    # Periodic spline
    y.append(y[0])
    cs = CubicSpline(hour_knots, y, bc_type='periodic')
    fine_values = cs(hour_fine)

    max_idx = np.argmax(fine_values)
    max_vol = fine_values[max_idx]
    peak_hour = hour_fine[max_idx]
    peak_time = time(int(peak_hour), int((peak_hour % 1)*60))

    results.append((location_id, direction, year, peak_time, max_vol))

df_results = pl.DataFrame(
    data=results,
    schema=["LocationID", "Direction", "Year", "PeakTime", "MaxVolume"],
    orient="row"
)

# Convert max_vol to int, convert PeakTime to "HH:MM" strings
df_results = df_results.with_columns([
    pl.col("MaxVolume").cast(pl.Int64),
    pl.Series(
        name="PeakTime",
        values=[f"{t.hour:02d}:{t.minute:02d}" for t in df_results["PeakTime"]]
    )
])

df_results = df_results.join(df_meta, on="LocationID", how="inner")

df_time = (
    df_results
    .select(["ATR_NAME", "LocationID", "Direction", "Year", "PeakTime"])
    .pivot(
        index=["ATR_NAME", "LocationID", "Direction"],
        on="Year",
        values="PeakTime"
    )
    .rename({str(y): f"{y}_Time" for y in keep_years})
)

df_vol = (
    df_results
    .select(["ATR_NAME", "LocationID", "Direction", "Year", "MaxVolume"])
    .pivot(
        index=["ATR_NAME", "LocationID", "Direction"],
        on="Year",
        values="MaxVolume"
    )
    .rename({str(y): f"{y}_Vol" for y in keep_years})
)

df_wide = df_time.join(df_vol, on=["ATR_NAME", "LocationID", "Direction"], how="inner")

grouped_df = (
    df_wide
    .group_by(["LocationID", "ATR_NAME"])
    .agg(
        pl.struct(pl.col("*").exclude(["LocationID", "ATR_NAME"])).alias("directions")
    )
)

row_strs = []

for row in grouped_df.iter_rows(named=True):
    location_id = row["LocationID"]
    location_name = row["ATR_NAME"]
    directions_list = []
    row_str = f"\\rowcolor{{white}}\\rule{{0pt}}{{4ex}}\n\\multirow{{2}}{{*}}{{{location_id}}} & \\multirow{{2}}{{*}}{{{location_name}}} &"

    for i, item in enumerate(row["directions"]):
        data_str = f'{item["Direction"]} & {item["2018_Time"]} & {item["2019_Time"]} & {item["2023_Time"]} & {item["2024_Time"]} & {item["2018_Vol"]} & {item["2019_Vol"]} & {item["2023_Vol"]} & {item["2024_Vol"]} \\\\'
        if i == 0:
            row_str = row_str + data_str
        else:
            row_str = row_str + "\n& & " + data_str

    print(row_str)
    row_strs.append(row_str)

#print("\n".join(row_strs))

