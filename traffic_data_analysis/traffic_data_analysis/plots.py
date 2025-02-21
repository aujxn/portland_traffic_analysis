from matplotlib import colormaps
from matplotlib.gridspec import GridSpec
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import logging
from traffic_data_analysis.config import PLOTS_DIR

logger = logging.getLogger(__name__)

def plot_hourly_quartiles_by_years(info, group: pl.DataFrame, df_meta: pl.DataFrame, show=False):
    # Setup plot parameters
    divisions = 15
    quartiles = np.linspace(0.01, 0.99, divisions * 2)
    fill_color = 'C0'

    # Compute quantile gradients
    q_values = np.zeros((divisions*2, 25))
    for hour, data in group.group_by("Hour"):
        quartile = np.quantile(data["Volume"].to_numpy(), quartiles)
        q_values[:, hour[0]] = quartile

        # Ensure first and last y-values match for periodic spline
        if hour[0] == 0:
            q_values[:, 24] = quartile

    # Generate splines for all quantiles
    hour_knots = np.arange(25)
    splines = []
    for i, data in enumerate(q_values):
        if np.all(np.isfinite(data)):
            splines.append(CubicSpline(hour_knots, data, bc_type='periodic'))
        else:
            print(f'Percentile: {quartiles[i]} has NaNs')
            print(data)
            exit()

    hour_fine = np.linspace(0, 24, 500)
    spline_values = [cs(hour_fine) for cs in splines]

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 2, width_ratios=[13, 1], wspace=0.1)
    ax = fig.add_subplot(gs[0])

    # Plot quartile gradient
    for i in range(divisions):
        alpha = min((i + 1) / divisions, 0.1)
        lower_curve = spline_values[i]
        upper_curve = spline_values[-(i+1)]
        ax.fill_between(hour_fine, lower_curve, upper_curve, alpha=alpha, color=fill_color)

    start = group["DateTime"].min().year
    end = group["DateTime"].max().year
    year_count = end - start + 1

    # Plot median for each year
    year_medians = group.group_by(["Hour", "Year"]).agg(pl.col("Volume").median().alias("median")).sort("Hour")
    cmap_name = 'Reds'
    cmap = colormaps[cmap_name]
    for i, year in enumerate(range(start, end + 1)):
        year_median = year_medians.filter(pl.col("Year") == year)
        y = year_median["median"].to_list()
        if len(y) == 0:
            continue
        y_24 = y[0]
        y.append(y_24)
        cs = CubicSpline(hour_knots, y, bc_type='periodic')
        ax.plot(hour_fine, cs(hour_fine), color=cmap((i+1) / year_count), lw=1)

    # Decorate plot
    ax.set_xticks(np.arange(25), ['12 AM'] + [f'{i} AM' for i in range(1,12)] + ['12 PM'] + [f'{i} PM' for i in range(1,12)] + ['12 AM'], rotation=45)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Cars / Hour')
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)

    # Plot colorbar for years
    ax2 = fig.add_subplot(gs[1])
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax2.imshow(gradient, aspect='auto', cmap=cmap_name)
    ax2.set_yticks(np.linspace(1, 255, min(year_count, 8)))
    ax2.set_yticklabels(np.linspace(start, end - 1, min(year_count, 8), dtype=int))
    ax2.yaxis.tick_right()
    ax2.set_xticks([])  # Remove x-axis

    # Get metadata for title
    location_id = info[0]
    meta_match = df_meta.filter(pl.col("LocationID") == location_id)
    row = meta_match.row(0) if len(meta_match) > 0 else ["Unknown"] * 3
    atr_name, desc = row[1:3]

    days = 'Weekend' if info[2] else 'Weekday'
    title = f'{location_id}: {atr_name}, ({info[1]} {days})\n{desc}'
    fig.suptitle(title)

    # Show plot if required
    if show:
        plt.show()

    # Save plot
    out_fname = f'{location_id}_{atr_name}_{info[1]}_{days}.png'.replace(' ', '-')
    out_path = PLOTS_DIR / out_fname
    plt.savefig(out_path, dpi=150)
    plt.close()

def missing_data_visual(df_traffic: pl.DataFrame, df_meta: pl.DataFrame):
    df_traffic = df_traffic.with_columns(pl.col("DateTime").dt.replace_time_zone(None))
    location_groups = df_traffic.group_by("LocationID")
    n_locs = df_traffic["LocationID"].n_unique()

    fig, axs = plt.subplots(n_locs, 1, figsize=(12, 18))
    for k, (location, group) in enumerate(location_groups):
        direction_groups = group.group_by("Direction")

        location = location[0]
        meta_match = df_meta.filter(pl.col("LocationID") == location)
        row = meta_match.row(0) if len(meta_match) > 0 else ["Unknown"]
        atr_name = row[1]

        min_date = group["DateTime"].min()
        max_date = group["DateTime"].max()
        ax = axs[k]

        labels = []
        bar_height = 0.15
        spacing = 1.1
        for j, (direction, group) in enumerate(direction_groups):
            weekly_bins = pl.date_range(min_date, max_date, "1w", eager=True)

            coverage = []
            total_expected = 0
            total_actual = 0
            for i in range(len(weekly_bins) - 1):
                start, end = weekly_bins[i], weekly_bins[i + 1]

                expected_hours = pl.datetime_range(start, end, interval="1h", eager=True)
                total_expected += len(expected_hours)

                actual_hours = group.filter((pl.col("DateTime") >= start) & (pl.col("DateTime") < end))["DateTime"]
                total_actual += len(actual_hours)

                coverage.append({"start": start, "end": end, "coverage": 1.0 - (len(actual_hours) / len(expected_hours))})

            coverage_df = pl.DataFrame(coverage)
            total_ratio = total_actual / total_expected
            percent = int(total_ratio * 100)
            label = f'{direction}: {percent}%'
            labels.append(label)

            for row in coverage_df.iter_rows(named=True):
                ax.barh(j * bar_height * spacing, width=(row['end'] - row['start']).days, left=row['start'], align='edge',
                        color='red', alpha=row['coverage'], height=bar_height)

        ax.set_yticks([bar_height / 2, bar_height * spacing + bar_height / 2], labels, rotation=45) 
        ax.set_title(f'{location}: {atr_name}', fontsize=16)
        ax.set_xlim(min_date, max_date)

    fig.tight_layout()
    plt.savefig(PLOTS_DIR / 'missing_data.png')
