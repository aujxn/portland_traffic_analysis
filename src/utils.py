import pandas as pd
import polars as pl
import os
import glob
import logging

logger = logging.getLogger(__name__)

# Here you can change the location of the data files if you like.
# This should probably live in a dotenv file if including keys or secrets, otherwise yaml config better
datafolder = '../data'
metadata_file = f'{datafolder}/ATR_metadata.csv'
processed_long_df_file = f'{datafolder}/processed_long.pq'

def load_raw_parquet() -> pl.DataFrame:
    return read_all_raw(datafolder)

def process_raw(out_file=processed_long_df_file) -> pd.DataFrame:
    df_polars = preprocess_data_polars(datafolder)
    logger.info('Successfully processed raw data to long format')
    # Convert back to pandas since heavy lifting is done
    df_final = df_polars.to_pandas()

    if out_file is not None:
        logger.info(f'Saving processed long dataframe to {out_file}')
        df_final.to_parquet(out_file, engine='pyarrow')

    return df_final

# Now we have a (long) Polars DataFrame with columns:
# [OrigLocationID, LocationID, Lane, Direction, DirectionRaw, ParsedDirection, DateTime, Volume, Year, Hour, Weekend]
# and each row corresponds to one hour of data at the ATR location
def load_processed_pandas() -> pd.DataFrame:
    if os.path.exists(processed_long_df_file):
        df = pl.read_parquet(processed_long_df_file).to_pandas()
        logger.info('Processed dataframe file loaded successfully')
    else:
        logger.warn("Couldn't find processed dataframe file... attempting to generate.")
        df = process_raw()
        
    # Unify types for consistency
    df['LocationID'] = df['LocationID'].astype('int64') 
    df= df.convert_dtypes()

    logger.info(df.dtypes.to_string())
    return df

def load_metadata_pandas() -> pd.DataFrame:
    if os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        logger.info('Metadata dataframe loaded successfully')

        # Select only columns we need so easier to work with
        useful_cols = ['SITE_ID', 'ATR_NAME', 'LOCATION', 'HWYNAME', 'HWYNUMB', 'COUNTYNAME', 'LAT', 'LONGTD']
        meta_df = meta_df[useful_cols]

        # Types and naming are inconsistent so fix that
        meta_df.rename(columns={"SITE_ID": "LocationID"}, inplace=True)
        meta_df = meta_df.convert_dtypes()

        logger.info(meta_df.dtypes.to_string())
        return meta_df

    logger.error(f'Metadata file not found, should be located at {metadata_file}')
    exit(-1)


# Helper function for loading the raw data
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
def read_all_raw(folder: str) -> pl.DataFrame:
    csv_files = glob.glob(f"{folder}/ATR26*.csv")
    pq_files  = glob.glob(f"{folder}/ATR*.pq")
    dfs = []

    for file in csv_files:
        df_csv = pl.read_csv(file, null_values=["NA", "Na", "na", "NaN", "nan", "None", "none", "Null", "null"])
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
    df = read_all_raw(folder)

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


