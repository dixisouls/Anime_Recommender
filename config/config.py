import torch


class Config:
    # Paths to data files
    ANIME_PATH = "data/anime.csv"
    RATINGS_PATH = "data/rating.csv"

    # Paths to save model and plots
    MODEL_SAVE_PATH = "saved_models/anime_recommender.pth"
    PLOT_SAVE_PATH = "plots/training_progress.png"

    # Training parameters
    BATCH_SIZE = 64  # Number of samples per training batch
    GAMMA = 0.95  # Discount factor for future rewards
    EPS_START = 1.0  # Initial epsilon value for epsilon-greedy policy
    EPS_END = 0.05  # Minimum epsilon value
    EPS_DECAY = 0.997  # Decay rate for epsilon
    TARGET_UPDATE = 5  # Frequency of target network updates
    NUM_EPISODES = 2000  # Number of episodes to train for
    MEMORY_CAPACITY = 10000  # Capacity of the replay memory

    # Model architecture parameters
    HIDDEN_SIZES = [1024, 512, 256]  # Sizes of hidden layers
    DROPOUT_RATE = 0.2  # Dropout rate for regularization

    # Environment parameters
    N_RECOMMENDATIONS = 5  # Number of recommendations to generate

    # Learning rate for the optimizer
    LEARNING_RATE = 0.0001

    # Device to run the model on (CPU or GPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed for reproducibility
    RANDOM_SEED = 42
