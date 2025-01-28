import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    @staticmethod
    def preprocess_data(
        anime_path: str, ratings_path: str
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Preprocess the anime and ratings data.

        Args:
            anime_path (str): Path to the anime dataset CSV file.
            ratings_path (str): Path to the ratings dataset CSV file.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed anime and ratings DataFrames.
        """
        try:
            # Load data from CSV files
            logger.info("Loading data from CSV files...")
            anime_df = pd.read_csv(anime_path)
            ratings_df = pd.read_csv(ratings_path)

            # Remove entries with missing values
            logger.info("Removing entries with missing values...")
            anime_df = anime_df.dropna()

            # Remove unwatched ratings (ratings with value -1)
            logger.info("Removing unwatched ratings...")
            ratings_df = ratings_df[ratings_df["rating"] != -1]

            # Filter for common anime IDs in both datasets
            logger.info("Filtering for common anime IDs...")
            common_anime_ids = set(anime_df["anime_id"]) & set(ratings_df["anime_id"])
            anime_df = anime_df[anime_df["anime_id"].isin(common_anime_ids)]
            ratings_df = ratings_df[ratings_df["anime_id"].isin(common_anime_ids)]

            logger.info("Data preprocessing complete!")
            return anime_df, ratings_df

        except Exception as e:
            # Log and raise any exceptions that occur during preprocessing
            logger.error(f"An error occurred during data preprocessing: {e}")
            raise e
