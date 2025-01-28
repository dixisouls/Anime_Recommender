import argparse
import logging
from config.config import Config
from data.data_preprocessor import DataPreprocessor
from environment.anime_env import AnimeEnvironment
from agents.dql_agent import DQLAgent
from utils.helpers import (
    setup_logging,
    plot_training_progress,
    create_save_directories,
    save_metadata,
)
import torch


def get_config_dict():
    """Convert Config class variables to a dictionary"""
    config_dict = {}
    for attr in dir(Config):
        # Skip private/special attributes
        if not attr.startswith("_"):
            value = getattr(Config, attr)
            # Convert torch.device to string
            if isinstance(value, torch.device):
                value = str(value)
            config_dict[attr] = value
    return config_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train Anime Recommender")
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
        "--num_episodes",
        type=int,
        default=Config.NUM_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=Config.MODEL_SAVE_PATH,
        help="Path to save trained model",
    )
    parser.add_argument(
        "--plot_save_path",
        type=str,
        default=Config.PLOT_SAVE_PATH,
        help="Path to save training plot",
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create save directories
    create_save_directories([args.model_save_path, args.plot_save_path])

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        anime_df, ratings_df = DataPreprocessor.preprocess_data(
            args.anime_path, args.ratings_path
        )

        # Create environment
        logger.info("Creating environment...")
        env = AnimeEnvironment(
            anime_df, ratings_df, n_recommendations=Config.N_RECOMMENDATIONS
        )

        # Initialize agent
        logger.info("Initializing agent...")
        state_size = len(env.animes)
        action_size = len(env.animes)
        agent = DQLAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=Config.HIDDEN_SIZES,
            device=Config.DEVICE,
            batch_size=Config.BATCH_SIZE,
            gamma=Config.GAMMA,
            eps_start=Config.EPS_START,
            eps_end=Config.EPS_END,
            eps_decay=Config.EPS_DECAY,
            target_update=Config.TARGET_UPDATE,
            memory_capacity=Config.MEMORY_CAPACITY,
            dropout_rate=Config.DROPOUT_RATE,
            learning_rate=Config.LEARNING_RATE,
        )

        # Train agent
        logger.info("Starting training...")
        rewards_history = agent.train(env, args.num_episodes)

        # Plot training progress
        logger.info("Plotting training progress...")
        plot_training_progress(rewards_history, args.plot_save_path)

        # Save model and metadata
        logger.info("Saving model and metadata...")
        agent.save(args.model_save_path)

        # Prepare metadata with JSON-serializable values
        metadata = {
            "state_size": state_size,
            "action_size": action_size,
            "num_episodes": args.num_episodes,
            "final_epsilon": float(agent.epsilon),  # Convert to native Python float
            "num_users": len(env.users),
            "num_animes": len(env.animes),
            "config": get_config_dict(),  # Use the helper function to get serializable config
        }

        save_metadata("saved_models/metadata.json", metadata)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
