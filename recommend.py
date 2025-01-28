import argparse
import logging
import json
from config.config import Config
from data.data_preprocessor import DataPreprocessor
from environment.anime_env import AnimeEnvironment
from agents.dql_agent import DQLAgent
from utils.helpers import setup_logging, get_recommendations
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Get Anime Recommendations")
    parser.add_argument(
        "--anime_path",
        type=str,
        default=Config.ANIME_PATH,
        help="Path to anime dataset",
    )
    parser.add_argument(
        "--ratings_path",
        type=str,
        default=Config.RATINGS_PATH,
        help="Path to ratings dataset",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=Config.MODEL_SAVE_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--num_recommendations",
        type=int,
        default=Config.N_RECOMMENDATIONS,
        help="Number of recommendations to generate",
    )
    return parser.parse_args()


def load_metadata(path: str = "saved_models/metadata.json"):
    """
    Load saved metadata from a JSON file.

    Args:
        path (str): Path to the metadata JSON file.

    Returns:
        dict: Loaded metadata.
    """
    with open(path, "r") as f:
        return json.load(f)


def format_recommendation_output(recommendations, anime_df):
    """
    Format and print recommendations with detailed information.

    Args:
        recommendations (list): List of recommended anime IDs and names.
        anime_df (pd.DataFrame): DataFrame containing anime information.
    """
    print("\nRecommended Animes:")
    print("=" * 80)

    for i, (anime_id, anime_name) in enumerate(recommendations, 1):
        anime_info = anime_df[anime_df["anime_id"] == anime_id].iloc[0]
        print(f"\n{i}. {anime_name}")
        print("-" * 40)
        print(f"Genre: {anime_info['genre']}")
        print(f"Type: {anime_info['type']}")
        print(f"Episodes: {anime_info['episodes']}")
        print(f"Rating: {anime_info['rating']:.2f}/10")
        print(f"Members: {anime_info['members']:,}")


def main():
    """
    Main function to generate anime recommendations for a user.
    """
    # Parse command-line arguments
    args = parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load metadata
        logger.info("Loading metadata...")
        metadata = load_metadata()

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        anime_df, ratings_df = DataPreprocessor.preprocess_data(
            args.anime_path, args.ratings_path
        )

        # Create environment
        logger.info("Creating environment...")
        env = AnimeEnvironment(
            anime_df, ratings_df, n_recommendations=args.num_recommendations
        )

        # Initialize agent
        logger.info("Initializing agent...")
        agent = DQLAgent(
            state_size=metadata["state_size"],
            action_size=metadata["action_size"],
            hidden_sizes=Config.HIDDEN_SIZES,
            device=Config.DEVICE,
        )

        # Load trained model
        logger.info("Loading trained model...")
        agent.load(args.model_path)

        while True:
            print("=" * 80)
            user_id = int(input("Enter user id: "))
            if user_id <= 0:
                logger.info("Exiting program...")
                break
            elif user_id not in env.users:
                print(f"User {user_id} not found in dataset. Please try again.")
                logger.error(f"User {user_id} not found in dataset")
            else:
                # Get recommendations
                logger.info(f"Generating recommendations for user {user_id}...")
                recommendations = get_recommendations(
                    agent, env, user_id, args.num_recommendations
                )

                # Format and display recommendations
                format_recommendation_output(recommendations, anime_df)

                logger.info("Recommendations generated successfully!")

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
