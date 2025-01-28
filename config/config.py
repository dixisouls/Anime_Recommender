import torch


class Config:
    # Data paths
    ANIME_PATH = "data/anime.csv"
    RATINGS_PATH = "data/rating.csv"

    # Model paths
    MODEL_SAVE_PATH = "saved_models/anime_recommender.pth"
    PLOT_SAVE_PATH = "plots/training_progress.png"

    # Training parameters
    BATCH_SIZE = 64
    GAMMA = 0.95
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.997
    TARGET_UPDATE = 5
    NUM_EPISODES = 2000
    MEMORY_CAPACITY = 10000

    # Model architecture
    HIDDEN_SIZES = [1024, 512, 256]
    DROPOUT_RATE = 0.2

    # Environment parameters
    N_RECOMMENDATIONS = 5

    # Learning rate
    LEARNING_RATE = 0.0001

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed
    RANDOM_SEED = 42
