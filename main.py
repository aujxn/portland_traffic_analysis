import pandas as pd
import polars as pl
import os
import glob
from polars import col
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    metadata_file = './data/ATR_metadata.csv'
    datafolder = './data/'

    # Now we have a (long) Polars DataFrame with columns:
    # [OrigLocationID, LocationID, Lane, Direction, DirectionRaw, ParsedDirection, DateTime, Volume, Year, Hour, Weekend]
    # and each row corresponds to one hour of data at the ATR location
    df_polars = preprocess_data_polars(datafolder)

    # Convert back to pandas since heavy lifting is done
    df_final = df_polars.to_pandas()

    # If you want to save the processed DataFrame back to a Parquet file, do:
    # df_final.to_parquet('./data/processed_ATR10_18-24.pq', engine='pyarrow')

    print(df_final.dtypes.to_string())
    print(df_final)

    # Load the ATR metadata into a dataframe
    meta_df = pd.read_csv(metadata_file)

    batch_plots(df_final, meta_df)

def parse_location_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Splits OrigLocationID into up to 3 underscore parts:
      - part0 = base_id
      - part1 = either lane or direction (if 2-part ID)
      - part2 = either direction (if 3-part ID) or None
    Derives:
      Lane: numeric if found in part1 or part2
      ParsedDirection: EB, WB, NB, SB if found in part1 or part2
    """

    df = df.with_columns(
        pl.col("OrigLocationID").str.splitn("_", 3).alias("id_parts")
    )

    # Extract each part (which might be None if fewer underscores)
    # part0 = base ID
    # part1 = maybe lane or direction
    # part2 = maybe direction
    df = df.with_columns([
        pl.col("id_parts").struct.field("field_0").alias("part0"),
        pl.col("id_parts").struct.field("field_1").alias("part1"),
        pl.col("id_parts").struct.field("field_2").alias("part2"),
    ])

    # Define sets/patterns:
    numeric_pattern = r"^\d+$"
    valid_dirs = ["EB", "WB", "NB", "SB"]

    # Derive Lane from either part1 or part2 if numeric
    #   We first check part1; if not numeric, we check part2.
    #   If neither is numeric, lane is None.
    df = df.with_columns(
        pl.when(pl.col("part1").str.contains(numeric_pattern))
          .then(pl.col("part1").cast(pl.Int64, strict=False))
        .when(pl.col("part2").str.contains(numeric_pattern))
          .then(pl.col("part2").cast(pl.Int64, strict=False))
        .otherwise(None)
        .alias("Lane")
    )

    # Derive a parsed direction from either part2 or part1 if in {EB,WB,NB,SB}.
    #   3-part IDs -> direction is usually part2
    #   2-part IDs -> direction is part1 if not numeric
    df = df.with_columns(
        pl.when(pl.col("part2").is_in(valid_dirs))
          .then(pl.col("part2"))
        .when(pl.col("part1").is_in(valid_dirs))
          .then(pl.col("part1"))
        .otherwise(None)
        .alias("ParsedDirection")
    )

    # base_id becomes the new "LocationID"
    df = df.with_columns(
        pl.col("part0").alias("LocationID")
    )

    return df.drop("id_parts")

# Normalizing everthing between the differences in how the csv and parquet are loaded 
# in polars is really ugly... But the speedup of polars is worth it.
def read_all(folder: str) -> pl.DataFrame:
    csv_files = glob.glob(f"{folder}ATR26*.csv")
    pq_files  = glob.glob(f"{folder}*.pq")
    dfs = []

    for file in csv_files:
        df_csv = pl.read_csv(file, null_values=["NA", "NaN", "nan", "None"])
        # Parse datetime if needed
        df_csv = df_csv.with_columns([
            pl.col("StartDate").str.strptime(pl.Datetime("us"), strict=False),
            pl.col("EndDate").str.strptime(pl.Datetime("us"), strict=False),
        ])
        df_csv = df_csv.with_columns([
            pl.col("StartDate").dt.replace_time_zone("UTC"),
            pl.col("EndDate").dt.replace_time_zone("UTC"),
        ])
        # Force all integer columns to Int64
        df_csv = df_csv.with_columns(
            [
                pl.col(col_name).cast(pl.Int64)
                for col_name, dtype in df_csv.schema.items()
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
            ]
        )
        # Now force all columns matching `HXX` or `HXX_#` to Int64
        df_csv = df_csv.with_columns(
            pl.col("^H\\d+(_\\d+)?$").cast(pl.Int64, strict=False)
        )

        # Also unify `Interval`, `Total`, etc. if needed:
        df_csv = df_csv.with_columns(
            pl.col("Interval").cast(pl.Int64, strict=False),
            pl.col("Total").cast(pl.Int64, strict=False)
        )
        dfs.append(df_csv)

    for file in pq_files:
        df_pq = pl.read_parquet(file)
        # Ensure datetime columns are UTC
        df_pq = df_pq.with_columns([
            pl.col("StartDate").dt.replace_time_zone("UTC"), 
            pl.col("EndDate").dt.replace_time_zone("UTC"),
        ])
        # Force numeric columns to Int64
        df_pq = df_pq.with_columns(
            [
                pl.col(col_name).cast(pl.Int64)
                for col_name, dtype in df_pq.schema.items()
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
            ]
        )
        dfs.append(df_pq)

    # Now all DataFrames in `dfs` have matching dtypes for shared columns
    df_all = pl.concat(dfs, how="vertical")
    return df_all

def preprocess_data_polars(folder: str) -> pl.DataFrame:
    """
    Example read + robust parse of ID + unpivot (2 or 3 part IDs).
    """

    # 1) Read Parquet and CSVs
    df = read_all(folder)

    # 2) Rename StartDate -> DateTime
    df = df.rename({"StartDate": "DateTime"})

    # 3) Rename original col => OrigLocationID
    df = df.rename({"LocationID": "OrigLocationID"})

    # 4) Parse ID thoroughly
    df = parse_location_id(df)

    # 5) If you have a raw "Direction" column that is untrustworthy, keep or rename:
    # e.g. rename it to DirectionRaw
    df = df.rename({"Direction": "DirectionRaw"})

    # 6) Now `ParsedDirection` should have EB/WB/NB/SB (if found),
    #    or None if numeric or 1-WAY, 2-WAY, etc.

    # You can decide how to unify:
    # Option A: override direction
    df = df.with_columns(
        pl.when(pl.col("ParsedDirection").is_not_null())
          .then(pl.col("ParsedDirection"))
          .otherwise(pl.col("DirectionRaw"))
          .alias("Direction")
    )

    # 7) Unpivot H01..H24
    hour_cols = [f"H{i:02d}" for i in range(1,25)]
    df_long = df.unpivot(
        index=["OrigLocationID", "LocationID", "Lane", "DirectionRaw", "ParsedDirection", "Direction", "DateTime"],
        on=hour_cols,
        variable_name="HourCol",
        value_name="Volume"
    )

    # 8) Extract hour offset from "Hxx" => int
    df_long = df_long.with_columns([
        (pl.col("HourCol").str.strip_prefix("H").cast(pl.Int64) - 1).alias("HourOffset")
    ])

    # Shift DateTime and convert to PST/PDT
    df_long = df_long.with_columns([
        (pl.col("DateTime") + pl.duration(hours="HourOffset")).alias("DateTime")
    ])
    df_long = df_long.with_columns([
        pl.col("DateTime").dt.convert_time_zone("America/Los_Angeles").alias("DateTime")
    ])

    # 9) Additional columns
    df_long = df_long.drop(["HourCol", "HourOffset"])
    df_long = df_long.with_columns([
        pl.col("DateTime").dt.year().alias("Year"),
        pl.col("DateTime").dt.hour().alias("Hour"),
        (pl.col("DateTime").dt.weekday() >= 5).alias("Weekend")
    ])

    return df_long

def batch_plots(df_final, meta_df):
    """
    For each location in df_final, produce up to two 2x2 figures:
      - NB/SB if the location has those directions
      - EB/WB if the location has those directions
    """
    # 1. Convert types if needed
    meta_df['SITE_ID'] = meta_df['SITE_ID'].astype(str).str.strip()
    df_final['LocationID'] = df_final['LocationID'].astype(str).str.strip()

    # 2. Filter out lane-only (Lane not null) and directionless (NaN or empty)
    df_dir = df_final[df_final['Lane'].isna()].copy()  # only no-lane
    df_dir = df_dir[df_dir['Direction'].notna()].copy()
    df_dir = df_dir[df_dir['Direction'].isin(['NB','SB','EB','WB'])].copy()

    # 3. Unique locations
    loc_ids = df_dir['LocationID'].unique()

    for loc_id in loc_ids:
        # Check which directions exist for this location
        dirs_for_loc = df_dir.loc[df_dir['LocationID'] == loc_id, 'Direction'].unique()

        # If NB or SB is in the list, plot them together
        if any(d in dirs_for_loc for d in ['NB','SB']):
            plot_hourly(df_dir, meta_df, loc_id, directions=('NB','SB'), output_dir='plots')

        # If EB or WB is in the list, plot them together
        if any(d in dirs_for_loc for d in ['EB','WB']):
            plot_hourly(df_dir, meta_df, loc_id, directions=('EB','WB'), output_dir='plots')

def plot_hourly(df, meta_df, location_id, directions, output_dir="plots"):
    """
    Creates a 2×2 plot for a single `location_id`, splitting
    rows = (Weekday, Weekend) and columns = (directions[0], directions[1]).

    directions: tuple of two direction strings, e.g. ("NB","SB") or ("EB","WB").
    """

    # 1) Metadata lookup
    meta_match = meta_df[meta_df['SITE_ID'] == location_id.lstrip('0')]
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
        print(f"No data for {location_id} with directions {directions}. Skipping.")
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

    print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    main()
