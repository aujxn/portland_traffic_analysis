import polars as pl
import glob

def convert_all() -> pl.DataFrame:
    csv_files = glob.glob(f"ATR26*.csv")

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
        df_csv = df_csv.with_columns([
            pl.col("StartDate").dt.convert_time_zone("America/Los_Angeles").alias("StartDate"),
            pl.col("EndDate").dt.convert_time_zone("America/Los_Angeles").alias("EndDate")
        ])

        df_csv.write_csv(f"local_time{file}")
        
convert_all()
