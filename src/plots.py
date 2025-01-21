import numpy as np
import os
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def batch_plots(long_df, meta_df):
    """
    For each location in df_final, produce up to two 2x2 figures:
      - NB/SB if the location has those directions
      - EB/WB if the location has those directions
    """
    # 2. Filter out lane-only (Lane not null) and directionless (NaN or empty)
    df_dir = long_df[long_df['Lane'].isna()].copy()  # only no-lane
    df_dir = df_dir[df_dir['Direction'].notna()].copy()
    df_dir = df_dir[df_dir['Direction'].isin(['NB','SB','EB','WB'])].copy()

    # 3. Unique locations
    loc_ids = df_dir['LocationID'].unique()

    for loc_id in loc_ids:
        # Check which directions exist for this location
        dirs_for_loc = df_dir.loc[df_dir['LocationID'] == loc_id, 'Direction'].unique()

        # If NB or SB is in the list, plot them together
        if any(d in dirs_for_loc for d in ['NB','SB']):
            plot_hourly(df_dir, meta_df, loc_id, directions=('NB','SB'))

        # If EB or WB is in the list, plot them together
        if any(d in dirs_for_loc for d in ['EB','WB']):
            plot_hourly(df_dir, meta_df, loc_id, directions=('EB','WB'))

def plot_hourly(df, meta_df, location_id, directions, output_dir="../plots"):
    """
    Creates a 2×2 plot for a single `location_id`, splitting
    rows = (Weekday, Weekend) and columns = (directions[0], directions[1]).

    directions: tuple of two direction strings, e.g. ("NB","SB") or ("EB","WB").
    """

    # 1) Metadata lookup
    meta_match = meta_df[meta_df['LocationID'] == location_id]
    if not meta_match.empty:
        row = meta_match.iloc[0]
        atr_name = str(row.get('ATR_NAME', 'Unknown')).strip()
        location_str = str(row.get('LOCATION', '')).strip()
    else:
        #atr_name = "Unknown"
        #location_str = ""
        return

    # 2) Filter main df for this location, these two directions only
    #    Also filter to keep years of interest
    keep_years = [2018, 2019, 2023, 2024]
    df_loc = df[
        (df['LocationID'] == location_id) &
        (df['Direction'].isin(directions)) &
        (df['Year'].isin(keep_years))
    ].copy()

    if df_loc.empty:
        logger.warn(f"No data for {location_id} with directions {directions}. Skipping.")
        return

    # 3) Create a 'Period' column (2018–2019 vs. 2023–2024) for plotting
    df_loc['Period'] = np.where(df_loc['Year'].isin([2018, 2019]),
                                '2018–2019', '2023–2024')

    # 4) Aggregate stats: mean, Q1, Q3, IQR by (Period, Weekend, Hour, Direction)
    stats = (
        df_loc.groupby(['Period', 'Weekend', 'Hour', 'Direction'])['Volume']
              .agg(
                  mean='mean',
                  Q1=lambda x: x.quantile(0.25),
                  Q3=lambda x: x.quantile(0.75)
              )
              .reset_index()
    )
    stats['IQR'] = stats['Q3'] - stats['Q1']

    # 5) Set up figure: 2x2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=False, sharey=True)
    (ax_wkday_dir1, ax_wkday_dir2), (ax_wkend_dir1, ax_wkend_dir2) = axes

    # We'll store subplots in a dict to simplify referencing
    # The top row is Weekday, bottom row is Weekend
    # The left col is directions[0], right col is directions[1]
    subplot_map = {
        (False, directions[0]): ax_wkday_dir1,
        (False, directions[1]): ax_wkday_dir2,
        (True,  directions[0]): ax_wkend_dir1,
        (True,  directions[1]): ax_wkend_dir2,
    }

    # Colors for the two Period categories
    period_colors = {
        '2018–2019': 'C0',
        '2023–2024': 'C1'
    }

    # 6) Plot each subset: Weekday vs Weekend, Direction 1 vs Direction 2
    for weekend_flag in [False, True]:
        for direction in directions:
            ax = subplot_map[(weekend_flag, direction)]
            # Subset data
            sub = stats[(stats['Weekend'] == weekend_flag) & (stats['Direction'] == direction)]
            # Plot each Period
            for period in ['2018–2019', '2023–2024']:
                sub_p = sub[sub['Period'] == period]
                if sub_p.empty:
                    continue
                color = period_colors[period]
                ax.plot(sub_p['Hour'], sub_p['mean'], color=color, label=period)
                ax.fill_between(sub_p['Hour'], sub_p['Q1'], sub_p['Q3'],
                                alpha=0.2, color=color)

            # Title for each subplot: e.g. "Weekday NB" or "Weekend SB"
            # We'll interpret weekend_flag == False => "Weekday", True => "Weekend"
            w_str = "Weekend" if weekend_flag else "Weekday"
            ax.set_title(f"{w_str} {direction}")

            # Add grid
            ax.grid(True, which='both', axis='both', alpha=0.3)

    # We'll add legends only in top-left subplot, or wherever you prefer
    ax_wkday_dir1.legend(loc='upper right')

    # 7) Shared x/y labels & ticks
    hours = np.arange(0, 24, 2)
    hour_labels = [f"{h:02d}:00" for h in hours]

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(hours)
            ax.set_xticklabels(hour_labels, rotation=45, ha='right')
            ax.set_xlim(0, 23)
            ax.set_xlabel("Time of Day")
            ax.set_ylabel("Volume")

    # 8) Two-line suptitle: first line => ID & ATR_NAME, second line => location
    fig.suptitle(f"ATR_{location_id} - {atr_name}\n{location_str}", fontsize=14)
    #fig.text(0.5, 0.94, location_str, ha='center', va='top', fontsize=10)
    plt.tight_layout()
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

    # 9) Save figure
    os.makedirs(output_dir, exist_ok=True)
    # We'll put directions in the filename
    direction_str = "_".join(directions)
    out_fname = f"{location_id}_{atr_name}_{direction_str}.png".replace(" ", "_")
    out_path = os.path.join(output_dir, out_fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved plot: {out_path}")
