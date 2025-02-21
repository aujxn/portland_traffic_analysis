import polars as pl
from traffic_data_analysis.utils import load_directional_hourly
from traffic_data_analysis.linear_models import build_gam_model, plot_gam_model

df_traffic, df_meta = load_directional_hourly()

keep_years = [2018, 2019, 2023, 2024]
location_id = 26024
direction = "SB"

df = df_traffic.filter(
    (pl.col("Year").is_in(keep_years)) & 
    (pl.col("LocationID") == location_id) & 
    (pl.col("Weekend") == False) & 
    (pl.col("Direction") == direction)
)

df = df.with_columns([
        "Year",
        "Hour",
        (pl.col("DateTime").dt.month()).alias("Month"),
        "Volume"
    ])

model, spline_hour, spline_month, n_knots_hour, n_knots_month = build_gam_model(df)
plot_gam_model(model, spline_hour, spline_month, n_knots_hour-1, n_knots_month-1)
