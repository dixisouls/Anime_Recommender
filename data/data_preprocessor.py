import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    @staticmethod
    def preprocess_data(
        anime_path: str, ratings_path: str
    ) -> (pd.DataFrame, pd.DataFrame):
        try:
            logger.info("Loading data from csv files...")
            anime_df = pd.read_csv(anime_path)
            ratings_df = pd.read_csv(ratings_path)

            logger.info("Removing entries with missing values...")
            anime_df = anime_df.dropna()

            logger.info("Removing unwatched ratings...")
            ratings_df = ratings_df[ratings_df["rating"] != -1]

            logger.info("Filtering for common anime Ids...")
            common_anime_ids = set(anime_df["anime_id"]) & set(ratings_df["anime_id"])
            anime_df = anime_df[anime_df["anime_id"].isin(common_anime_ids)]
            ratings_df = ratings_df[ratings_df["anime_id"].isin(common_anime_ids)]

            logger.info("Data preprocessing complete!")
            return anime_df, ratings_df

        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {e}")
            raise e
