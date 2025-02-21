import polars as pl
from traffic_data_analysis.config import DATA_DIR, DATA_FILE

import logging
logger = logging.getLogger(__name__)

PROCESSED_LONG_DF_FILE = DATA_DIR / "processed_long.pq"
DIRECTIONAL_HOURLY_DF_FILE = DATA_DIR / "directional_long.pq"
FILTERED_META_DF_FILE = DATA_DIR / "filtered_meta.pq"

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

    # extract each part (which might be None if fewer underscores)
    #   part0 = base ID
    #   part1 = maybe lane or direction
    #   part2 = maybe direction
    df = df.with_columns([
        pl.col("id_parts").struct.field("field_0").alias("part0"),
        pl.col("id_parts").struct.field("field_1").alias("part1"),
        pl.col("id_parts").struct.field("field_2").alias("part2"),
    ])

    # define sets/patterns:
    numeric_pattern = r"^\d+$"
    valid_dirs = ["EB", "WB", "NB", "SB"]

    # derive Lane from either part1 or part2 if numeric
    #   we first check part1; if not numeric, we check part2.
    #   if neither is numeric, lane is None.
    df = df.with_columns(
        pl.when(pl.col("part1").str.contains(numeric_pattern))
          .then(pl.col("part1").cast(pl.Int64, strict=False))
        .when(pl.col("part2").str.contains(numeric_pattern))
          .then(pl.col("part2").cast(pl.Int64, strict=False))
        .otherwise(None)
        .alias("Lane")
    )

    # derive a parsed direction from either part2 or part1 if in {EB,WB,NB,SB}.
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

    return df.drop("id_parts").cast({"LocationID": pl.Int64})

def preprocess() -> pl.DataFrame:
    """
    Converts the raw data into long format for easier analysis and saves the processed DataFrame.
    """
    df_long = wrangle()
    logger.info('Successfully processed raw data to long format')

    logger.info(f'Saving processed long dataframe to {PROCESSED_LONG_DF_FILE}')
    df_long.write_parquet(PROCESSED_LONG_DF_FILE)

    return df_long

def wrangle() -> pl.DataFrame:
    """
    Reads data file, robust parse of inconsistent ID names (2 or 3 part IDs), and unpivot into long.
    """

    df = pl.read_parquet(DATA_FILE)
    df = df.rename({"StartDate": "DateTime"})
    # the raw data has additional data in the `LocationID` column,
    # so let's rename it and parse out a consistent naming
    df = df.rename({"LocationID": "OrigLocationID"})
    df = parse_location_id(df)
    # the direction might also be specficed in this column, rename it
    # and then use the direction we parsed from the `LocationID`
    df = df.rename({"Direction": "DirectionRaw"})
    # when parsing the direction fails, fallback to the original direction
    df = df.with_columns(
        pl.when(pl.col("ParsedDirection").is_not_null())
          .then(pl.col("ParsedDirection"))
          .otherwise(pl.col("DirectionRaw"))
          .alias("Direction")
    )

    # TODO should extract 15 minute data also and pivot to seperate DataFrame
    # unpivot H01..H24 into long format
    hour_cols = [f"H{i:02d}" for i in range(1,25)]
    df_long = df.unpivot(
        index=["OrigLocationID", "LocationID", "Lane", "DirectionRaw", "ParsedDirection", "Direction", "DateTime"],
        on=hour_cols,
        variable_name="HourCol",
        value_name="Volume"
    )

    # extract hour offset from "Hxx" => int
    df_long = df_long.with_columns([
        (pl.col("HourCol").str.strip_prefix("H").cast(pl.Int64) - 1).alias("HourOffset")
    ])

    # shift DateTime and convert to PST/PDT
    df_long = df_long.with_columns([
        (pl.col("DateTime") + pl.duration(hours="HourOffset")).alias("DateTime")
    ])
    df_long = df_long.with_columns([
        pl.col("DateTime").dt.convert_time_zone("America/Los_Angeles").alias("DateTime")
    ])

    # add additional columns and drop intermediate ones
    df_long = df_long.drop(["HourCol", "HourOffset"])
    df_long = df_long.with_columns([
        pl.col("DateTime").dt.year().alias("Year"),
        pl.col("DateTime").dt.hour().alias("Hour"),
        (pl.col("DateTime").dt.weekday() >= 5).alias("Weekend")
    ])

    return df_long
