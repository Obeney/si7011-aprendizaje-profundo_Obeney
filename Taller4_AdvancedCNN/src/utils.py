"""Constantes compartidas — TweetEval emotion."""

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LABEL_NAMES = ["anger", "joy", "optimism", "sadness"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)
HF_DATASET = ("tweet_eval", "emotion")
