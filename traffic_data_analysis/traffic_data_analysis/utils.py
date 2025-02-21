import polars as pl
import os
import hashlib
import urllib.request
import logging
from traffic_data_analysis.config import META_FILE, CHECKSUMS_URL, DATA_SERVER_URL
from traffic_data_analysis.preprocess import preprocess, PROCESSED_LONG_DF_FILE, DIRECTIONAL_HOURLY_DF_FILE, FILTERED_META_DF_FILE

logger = logging.getLogger(__name__)

# Now we have a (long) Polars DataFrame with columns:
# [OrigLocationID, LocationID, Lane, Direction, DirectionRaw, ParsedDirection, DateTime, Volume, Year, Hour, Weekend]
# and each row corresponds to one hour of data at the ATR location
def load_processed_long() -> pl.DataFrame:
    if os.path.exists(PROCESSED_LONG_DF_FILE):
        df = pl.read_parquet(PROCESSED_LONG_DF_FILE)
        logger.info('Processed dataframe file loaded successfully')
    else:
        logger.warning("Couldn't find processed dataframe file... attempting to generate.")
        df = preprocess()
        
    logger.info(df.schema)
    return df

def load_metadata() -> pl.DataFrame:
    if os.path.exists(META_FILE):
        meta_df = pl.read_csv(META_FILE)
        logger.info('Metadata dataframe loaded successfully')

        # Select only columns we need
        useful_cols = ['SITE_ID', 'ATR_NAME', 'LOCATION', 'HWYNAME', 'HWYNUMB', 'COUNTYNAME', 'LAT', 'LONGTD']
        meta_df = meta_df.select(useful_cols)

        # Fix column names and types
        meta_df = meta_df.rename({"SITE_ID": "LocationID"}).cast({"LocationID": pl.Int64})

        logger.info(meta_df.schema)
        return meta_df

    logger.error(f'Metadata file not found, should be located at {META_FILE}\nTry downloading by running `fetch_data`.')
    exit(-1)

def load_directional_hourly() -> tuple[pl.DataFrame, pl.DataFrame]:
    if os.path.exists(DIRECTIONAL_HOURLY_DF_FILE) and os.path.exists(FILTERED_META_DF_FILE):
        df_traffic = pl.read_parquet(DIRECTIONAL_HOURLY_DF_FILE)
        df_meta = pl.read_parquet(FILTERED_META_DF_FILE)
        logger.info('Processed dataframe file loaded successfully')
    else:
        logger.warning("Couldn't find directional hourly dataframe file... attempting to generate.")
        df_traffic = load_processed_long()
        df_traffic = df_traffic.drop_nulls(["Volume"])  # Drop NaNs properly
        df_meta = load_metadata()

        # Merge metadata with traffic data
        merged = df_traffic.join(df_meta, on="LocationID", how="inner")

        # Keep only valid LocationIDs
        valid_ids = merged["LocationID"].unique()
        df_meta = df_meta.filter(df_meta["LocationID"].is_in(valid_ids))

        # Filter traffic data
        valid_dirs = ["NB", "SB", "EB", "WB"]
        df_traffic = df_traffic.filter(
            df_traffic["LocationID"].is_in(valid_ids) &
            df_traffic["Direction"].is_in(valid_dirs) &
            df_traffic["Lane"].is_null()
        )

        # Compute min and max DateTime per LocationID
        date_ranges = df_traffic.group_by("LocationID").agg(
            [
                pl.col("DateTime").min().alias("min_date"),
                pl.col("DateTime").max().alias("max_date")
            ]
        )

        # Merge date ranges into metadata
        df_meta = df_meta.join(date_ranges, on="LocationID", how="left")

        # Remove duplicate timestamps (caused by DST)
        df_traffic = df_traffic.unique(subset=["LocationID", "DateTime", "Direction"], keep="first")

        # Save results
        df_traffic.write_parquet(DIRECTIONAL_HOURLY_DF_FILE)
        df_meta.write_parquet(FILTERED_META_DF_FILE)
        logger.info('Directional hourly dataframes generated successfully')

    logger.info(df_traffic.schema)
    logger.info(df_meta.schema)
    return df_traffic, df_meta

def compute_md5(file_path):
    """Compute the MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def fetch_remote_checksums():
    """Download and parse the remote checksum file."""
    try:
        with urllib.request.urlopen(CHECKSUMS_URL) as response:
            lines = response.read().decode().splitlines()
        return {line.split()[0]: line.split()[1] for line in lines}
    except Exception as e:
        print(f"⚠ Warning: Could not fetch checksum file: {e}")
        return {}

def download_data():
    """Download missing or outdated data files from the remote server."""
    remote_checksums = fetch_remote_checksums()

    for file in DATA_FILES:
        local_path = DATA_DIR / file
        remote_checksum = remote_checksums.get(file)

        # If file exists, check if the checksum matches
        if local_path.exists():
            local_checksum = compute_md5(local_path)
            if remote_checksum and local_checksum == remote_checksum:
                print(f"✔ {file} is up to date.")
                continue  # Skip download if unchanged

        # Download the file if missing or outdated
        print(f"⬇ Downloading {file}...")
        try:
            urllib.request.urlretrieve(DATA_SERVER_URL + file, local_path)
            print(f"✔ Downloaded {file} to {local_path}")
        except Exception as e:
            print(f"❌ Failed to download {file}: {e}")
