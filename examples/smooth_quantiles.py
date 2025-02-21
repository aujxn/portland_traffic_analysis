import polars as pl
from traffic_data_analysis.plots import plot_hourly_quartiles_by_years, missing_data_visual
from traffic_data_analysis.utils import load_directional_hourly

def all_yearly_plots(df_traffic: pl.DataFrame, df_meta: pl.DataFrame):
    grouped = df_traffic.group_by(['LocationID', 'Direction', 'Weekend'])
    for info, group in grouped:
        print(f'Group: {info}')
        plot_hourly_quartiles_by_years(info, group, df_meta)

df_traffic, df_meta = load_directional_hourly()
all_yearly_plots(df_traffic, df_meta)
missing_data_visual(df_traffic, df_meta)
