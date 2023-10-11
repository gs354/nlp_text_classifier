from pathlib import Path
from nlp.dataset import make_batch_dataset, vectorize_text, preprocess_datasets
import numpy as np
import math

# Data paths
DATA_DIR = "./data"
TEST_DATA = Path(f"{DATA_DIR}/imdb_data/test")
TEST_DATA_NEG = Path(f"{DATA_DIR}/imdb_data/test/neg")
TEST_DATA_POS = Path(f"{DATA_DIR}/imdb_data/test/pos")

# Parameters
BATCH_SIZE = 32
NGRAMS = 2
MAX_TOKENS = 20000
OUTPUT_MODE = "multi_hot"


def test_make_batch_dataset() -> None:
    negative_files = TEST_DATA_NEG.rglob("*.txt")
    positive_files = TEST_DATA_POS.rglob("*.txt")

    _, counts_neg = np.unique([x.parent for x in negative_files], return_counts=True)
    _, counts_pos = np.unique([x.parent for x in positive_files], return_counts=True)

    dataset = make_batch_dataset(TEST_DATA, BATCH_SIZE)

    n_batches = len(list(dataset.as_numpy_iterator()))
    assert n_batches == int(math.ceil((counts_pos[0] + counts_neg[0]) / BATCH_SIZE))

    text_vectorization = vectorize_text(
        dataset, ngrams=NGRAMS, max_tokens=MAX_TOKENS, output_mode=OUTPUT_MODE
    )

    dataset = preprocess_datasets(dataset, text_vectorization, num_parallel_calls=4)

    count = 0
    for inputs, targets in dataset:
        assert inputs.shape == (BATCH_SIZE, MAX_TOKENS)
        assert targets.shape == (BATCH_SIZE,)
        count += 1
        if count == n_batches - 1:
            break
