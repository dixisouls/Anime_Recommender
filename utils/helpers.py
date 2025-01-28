import logging
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import os
import json
import pandas as pd


def setup_logging(log_path: str = "logs"):
    """Setup logging configuration"""
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_path}/anime_recommender.log"),
            logging.StreamHandler(),
        ],
    )


def get_recommendations(
    agent, env, user_id: int, n_recommendations: int = 5
) -> List[Tuple[int, str]]:
    """
    Get recommendations for a specific user

    Args:
        agent: Trained DQL agent
        env: Anime environment
        user_id (int): User ID
        n_recommendations (int): Number of recommendations to make

    Returns:
        List[Tuple[int, str]]: List of (anime_id, anime_name) recommendations
    """
    state = env.reset(user_id)
    recommendations = []

    for _ in range(n_recommendations):
        valid_actions = env.get_valid_actions(state)
        action = agent.select_action(state, valid_actions)
        anime_id = env.animes[action]
        anime_name = env.anime_df[env.anime_df["anime_id"] == anime_id]["name"].iloc[0]
        recommendations.append((anime_id, anime_name))
        state, _, _ = env.step(action, user_id)

    return recommendations


def plot_training_progress(rewards: List[float], save_path: str):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.6, label="Raw Rewards")

    # Add moving average
    window_size = 50
    moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
    plt.plot(moving_avg, linewidth=2, label=f"{window_size}-Episode Moving Average")

    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def create_save_directories(paths: List[str]):
    """Create necessary directories for saving files"""
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def save_metadata(path: str, metadata: Dict[str, Any]):
    """Save metadata and configuration"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
