from plots import batch_plots
from utils import load_metadata_pandas, load_processed_pandas
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    df = load_processed_pandas()
    logger.info(df)

    meta_df = load_metadata_pandas()
    logger.info(meta_df)

    batch_plots(df, meta_df)

if __name__ == "__main__":
    main()
