import polars as pl
import numpy as np
import calendar
import matplotlib.pyplot as plt
from datetime import time
from traffic_data_analysis.config import PLOTS_DIR
from traffic_data_analysis.preprocess import filter_bad_directional
from traffic_data_analysis.utils import load_directional_hourly
from traffic_data_analysis.linear_models import build_gam_model, plot_gam_model

def build_subfigure(files):
    subfig = lambda filename : "\\begin{{subfigure}}[b]{{0.45\\textwidth}}\n\t\t\\includegraphics[width=\\textwidth]{{{}}}\n\t\\end{{subfigure}}".format(filename)

    return f"\\begin{{figure}}[H]\n\t\\centering\n\t{subfig(files[0])}\n\t\\hfill\n\t{subfig(files[1])}\n\n\t{subfig(files[2])}\n\t\\hfill\n\t{subfig(files[3])}\n\\end{{figure}}"

def gam_all_locations(df_traffic: pl.DataFrame, df_meta: pl.DataFrame):
    keep_years = [2018, 2019, 2023, 2024]

    df_traffic = df_traffic.filter((pl.col("Year").is_in(keep_years)))

    grouped = df_traffic.group_by(['LocationID', 'Direction', 'Weekend'])
    files = []
    for info, group in grouped:
        group = group.with_columns([
            "Year",
            "Hour",
            (pl.col("DateTime").dt.month()).alias("Month"),
            "Volume"
        ])

        filename = gam_and_plot(info, group, df_meta)
        files.append(filename)
    return files

def gam_and_plot(info, group, df_meta):
    model, spline_hour, spline_interaction, spline_month, n_knots_hour, n_knots_interaction, n_knots_month = build_gam_model(group)

    w_hour = model.coef_[:n_knots_hour-1]
    rest = model.coef_[n_knots_hour-1:]
    w_interaction = rest[:n_knots_interaction-2]
    w_month = rest[n_knots_interaction-2:]

    interaction_cov = model.sigma_[n_knots_hour-1:,n_knots_hour-1:]
    interaction_cov = interaction_cov[:n_knots_interaction-2,:n_knots_interaction-2]

    #month_cov = model.sigma_[n_knots_interaction-2:,n_knots_interaction-2:]
    #month_cov = model.sigma_[:n_knots_interaction-2,:n_knots_interaction-2]
    dense_hours = np.linspace(0, 24, 100).reshape(-1, 1)

    X_hour = spline_hour.transform(dense_hours)
    base_hour = np.dot(X_hour, w_hour)

    X_hour_pd = spline_interaction.transform(dense_hours)
    interaction_effect = np.dot(X_hour_pd, w_interaction)
    #interaction_variance = X_hour_pd @ interaction_cov @ X_hour_pd.T
    interaction_stddev = np.sum(np.matmul(X_hour_pd, interaction_cov) * X_hour_pd, axis=1) ** 0.5

    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    pre_max_idx = np.argmax(base_hour)

    post_hour = base_hour + interaction_effect
    post_max_idx = np.argmax(post_hour)

    x = dense_hours[pre_max_idx, 0]
    y = base_hour[pre_max_idx]
    peak_time = time(int(x), int((x % 1)*60))
    label = f"{peak_time.hour:02d}:{peak_time.minute:02d}, {int(y)}"
    axs[0].plot(x, y, marker='o', linestyle='none', color='C0', label=label)
    #axs[0].annotate(f"{peak_time.hour:02d}:{peak_time.minute:02d}", (x, y), textcoords="offset points", xytext=(6,6), ha='center')

    x = dense_hours[post_max_idx, 0]
    y = post_hour[post_max_idx]
    peak_time = time(int(x), int((x % 1)*60))
    label = f"{peak_time.hour:02d}:{peak_time.minute:02d}, {int(y)}"
    axs[0].plot(x, y, marker='o', linestyle='none', color='C1', label=label)
    #axs[0].annotate(f"{peak_time.hour:02d}:{peak_time.minute:02d}", (x, y), textcoords="offset points", xytext=(6,6), ha='center')

    axs[0].plot(dense_hours, base_hour, color="C0", label="Pre-Covid, $s_1$")
    axs[0].plot(dense_hours, base_hour + interaction_effect, color="C1", label="Post-Covid, $s_1 + s_2$")
    axs[0].set_title("Hourly Traffic (monthly adjusted)")
    axs[0].legend()
    axs[0].set_xticks(np.arange(25), ['12 AM'] + [f'{i} AM' for i in range(1,12)] + ['12 PM'] + [f'{i} PM' for i in range(1,12)] + ['12 AM'], rotation=45)
    axs[0].set_xlabel('Time of Day')
    axs[0].set_ylabel('Cars / Hour')
    axs[0].set_xlim(0, 24)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(dense_hours, interaction_effect, label='$s_2$', color='darkred')
    lower_curve = interaction_effect - interaction_stddev*2.
    upper_curve = interaction_effect + interaction_stddev*2.
    axs[1].fill_between(dense_hours.flatten(), lower_curve, upper_curve, alpha=0.2, color='darkred', label='95% confidence')
    axs[1].axhline(y=0.0, color='b', linestyle='--')
    axs[1].set_title("Partial Dependence: Interaction Term (Post-Covid Hourly Effect)")
    axs[1].legend()
    axs[1].set_xticks(np.arange(25), ['12 AM'] + [f'{i} AM' for i in range(1,12)] + ['12 PM'] + [f'{i} PM' for i in range(1,12)] + ['12 AM'], rotation=45)
    axs[1].set_xlabel('Time of Day')
    axs[1].set_ylabel('Effect on Volume')
    axs[1].set_xlim(0, 24)
    axs[1].grid(True, alpha=0.3)

    dense_months = np.linspace(0, 12, 100).reshape(-1, 1)
    X_month = spline_month.transform(dense_months)
    monthly_correction = np.dot(X_month, w_month)
    axs[2].plot(dense_months, monthly_correction, label="$s_3$", color='darkred')
    axs[2].axhline(y=0.0, color='b', linestyle='--')
    axs[2].set_title("Partial Dependence: Monthly")
    axs[2].legend()
    axs[2].set_xticks(np.arange(13), [calendar.month_abbr[i] for i in range(1, 13)] + [calendar.month_abbr[1]], rotation=45)
    axs[2].set_xlabel('Month of Year')
    axs[2].set_ylabel('Effect on Volume')
    axs[2].set_xlim(0, 12)
    axs[2].grid(True, alpha=0.3)

    location_id = info[0]
    meta_match = df_meta.filter(pl.col("LocationID") == location_id)
    row = meta_match.row(0) if len(meta_match) > 0 else ["Unknown"] * 3
    atr_name, desc = row[1:3]
    days = 'Weekend' if info[2] else 'Weekday'
    title = f'{location_id}: {atr_name}, ({info[1]} {days})\n{desc}'
    fig.suptitle(title)
    fig.tight_layout()

    out_fname = f'{location_id}_{atr_name}_{info[1]}_{days}_gam.png'.replace(' ', '-')
    out_path = PLOTS_DIR / out_fname
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_fname

df_traffic, df_meta = load_directional_hourly()
df_traffic = filter_bad_directional(df_traffic)
files = gam_all_locations(df_traffic, df_meta)
files.sort()
print("\n\n".join([build_subfigure(files[i:i+4]) for i in range(0, len(files), 4)]))
    
