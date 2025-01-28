import pandas as pd
import torch
import numpy as np
from typing import Tuple, Dict
import logging
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class AnimeEnvironment:
    def __init__(
        self,
        anime_data: pd.DataFrame,
        ratings_data: pd.DataFrame,
        n_recommendations: int = 5,
    ):
        """
        Initialize the AnimeEnvironment.

        Args:
            anime_data (pd.DataFrame): DataFrame containing anime information.
            ratings_data (pd.DataFrame): DataFrame containing user ratings.
            n_recommendations (int): Number of recommendations to generate.
        """
        self.anime_df = anime_data
        self.ratings_df = ratings_data
        self.n_recommendations = n_recommendations

        # Create user-anime interaction matrix
        self.user_anime_matrix = pd.pivot_table(
            ratings_data,
            values="rating",
            index="user_id",
            columns="anime_id",
            fill_value=0,
        )

        # Calculate anime statistics
        self.anime_stats = self._calculate_anime_stats()

        self.users = self.user_anime_matrix.index.tolist()
        self.animes = self.user_anime_matrix.columns.tolist()

        # Create mappings from user/anime IDs to indices
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        self.anime_to_idx = {anime: idx for idx, anime in enumerate(self.animes)}

        # Convert ratings matrix to torch tensor
        self.ratings_matrix = torch.FloatTensor(self.user_anime_matrix.values)

        # Calculate user preferences
        self.user_preferences = self._calculate_user_preferences()

        logger.info(
            f"Environment initialized with {len(self.users)} users and {len(self.animes)} animes"
        )

    def _calculate_anime_stats(self) -> pd.DataFrame:
        """
        Calculate popularity and average ratings for each anime.

        Returns:
            pd.DataFrame: DataFrame containing anime statistics.
        """
        stats = pd.DataFrame()
        stats["popularity"] = self.ratings_df.groupby("anime_id").size()
        stats["avg_rating"] = self.ratings_df.groupby("anime_id")["rating"].mean()
        stats["rating_std"] = (
            self.ratings_df.groupby("anime_id")["rating"].std().fillna(0)
        )
        return stats

    def _calculate_user_preferences(self) -> Dict:
        """
        Calculate genre and type preferences for each user.

        Returns:
            Dict: Dictionary containing user preferences.
        """
        file_path = "saved_models/user_preferences.pkl"
        try:
            with open(file_path, "rb") as f:
                preferences = pickle.load(f)
            logger.info("User preferences loaded from file")
        except FileNotFoundError:
            preferences = {}
            total_users = len(self.users)

            logger.info(f"Starting to calculate preferences for {total_users} users...")

            # Create progress bar
            progress_bar = tqdm(
                total=total_users,
                desc="Calculating user preferences",
                ncols=100,
                position=0,
                leave=True,
            )

            # Track some statistics
            processed_ratings = 0
            unique_genres = set()
            unique_types = set()

            for user_id in self.users:
                # Get user's watched animes
                user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
                watched_animes = user_ratings.merge(self.anime_df, on="anime_id")

                processed_ratings += len(watched_animes)

                # Calculate genre preferences
                genre_ratings = {}
                for _, anime in watched_animes.iterrows():
                    genres = str(anime["genre"]).split(",")
                    rating = anime["rating_x"]
                    for genre in genres:
                        genre = genre.strip()
                        unique_genres.add(genre)
                        if genre not in genre_ratings:
                            genre_ratings[genre] = []
                        genre_ratings[genre].append(rating)

                # Average rating per genre
                genre_preferences = {
                    genre: np.mean(ratings) for genre, ratings in genre_ratings.items()
                }

                # Calculate type preferences
                type_ratings = (
                    watched_animes.groupby("type")["rating_x"].mean().to_dict()
                )
                unique_types.update(type_ratings.keys())

                preferences[user_id] = {
                    "genres": genre_preferences,
                    "types": type_ratings,
                }

                # Update progress bar and description
                progress_bar.update(1)
                if len(preferences) % 1000 == 0:  # Log every 1000 users
                    progress_bar.set_description(
                        f"Users: {len(preferences)}/{total_users} | "
                        f"Ratings: {processed_ratings:,} | "
                        f"Genres: {len(unique_genres)} | "
                        f"Types: {len(unique_types)}"
                    )

            # Close progress bar
            progress_bar.close()

            # Log final statistics
            logger.info(f"Preference calculation completed:")
            logger.info(f"- Total users processed: {total_users:,}")
            logger.info(f"- Total ratings processed: {processed_ratings:,}")
            logger.info(f"- Unique genres found: {len(unique_genres)}")
            logger.info(f"- Unique types found: {len(unique_types)}")
            logger.info(
                f"- Average ratings per user: {processed_ratings / total_users:.1f}"
            )

            # Save preferences to file
            with open(file_path, "wb") as f:
                pickle.dump(preferences, f)

        return preferences

    def reset(self, user_id: int) -> torch.Tensor:
        """
        Reset the environment for a new user session.

        Args:
            user_id (int): ID of the user.

        Returns:
            torch.Tensor: Initial state for the user.
        """
        self.current_user = user_id
        self.user_history = []
        self.current_state = self.get_user_state(user_id)
        return self.current_state

    def get_user_state(self, user_id: int) -> torch.Tensor:
        """
        Get the current state representation for a user.

        Args:
            user_id (int): ID of the user.

        Returns:
            torch.Tensor: State tensor for the user.
        """
        user_idx = self.user_to_idx[user_id]
        return self.ratings_matrix[user_idx]

    def _calculate_reward(
        self, user_id: int, anime_id: int, actual_rating: float
    ) -> float:
        """
        Calculate the reward for recommending an anime to a user.

        Args:
            user_id (int): ID of the user.
            anime_id (int): ID of the recommended anime.
            actual_rating (float): Actual rating given by the user.

        Returns:
            float: Calculated reward.
        """
        reward = 0.0

        # Base reward from actual rating
        if actual_rating > 0:
            reward += actual_rating / 10.0  # Normalize to [0,1]

        # Get anime information
        anime_info = self.anime_df[self.anime_df["anime_id"] == anime_id].iloc[0]

        # Genre matching reward
        user_prefs = self.user_preferences[user_id]["genres"]
        anime_genres = str(anime_info["genre"]).split(",")
        genre_reward = 0
        for genre in anime_genres:
            genre = genre.strip()
            if genre in user_prefs:
                genre_reward += user_prefs[genre] / 10.0
        reward += genre_reward / len(anime_genres) if anime_genres else 0

        # Popularity and rating reward
        stats = self.anime_stats.loc[anime_id]
        popularity_percentile = (
            stats["popularity"] / self.anime_stats["popularity"].max()
        )
        if stats["avg_rating"] > 7.0 and popularity_percentile < 0.5:
            reward += 0.2  # Bonus for discovering hidden gems

        # Diversity reward
        if self.user_history:
            previous_genres = set()
            for prev_anime in self.user_history:
                prev_info = self.anime_df[
                    self.anime_df["anime_id"] == self.animes[prev_anime]
                ].iloc[0]
                previous_genres.update(str(prev_info["genre"]).split(","))

            new_genres = set(anime_genres) - previous_genres
            if new_genres:
                reward += 0.1 * len(new_genres)  # Bonus for new genres

        return reward

    def step(self, action: int, user_id: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Execute an action and return the new state, reward, and done flag.

        Args:
            action (int): Index of the selected anime.
            user_id (int): ID of the user.

        Returns:
            Tuple[torch.Tensor, float, bool]: New state, reward, and done flag.
        """
        user_idx = self.user_to_idx[user_id]
        anime_id = self.animes[action]

        # Get actual rating
        actual_rating = self.ratings_matrix[user_idx][action].item()

        # Calculate reward
        reward = self._calculate_reward(user_id, anime_id, actual_rating)

        # Update state
        self.user_history.append(action)
        new_state = self.get_user_state(user_id)

        # Check if episode is done
        done = len(self.user_history) >= self.n_recommendations

        return new_state, reward, done

    def get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get a mask of valid actions (unwatched animes).

        Args:
            state (torch.Tensor): Current state tensor.

        Returns:
            torch.Tensor: Mask of valid actions.
        """
        return state == 0
