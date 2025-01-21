from utils import load_metadata_pandas, load_processed_pandas
import pandas as pd


df_traffic = load_processed_pandas()
df_meta = load_metadata_pandas()

df = pd.merge(df_traffic, df_meta, on='LocationID', how='inner')
print(df['LocationID'].unique())
print(df)

location_id = 26024
sub = df[df["LocationID"] == location_id].copy()
print(sub)

