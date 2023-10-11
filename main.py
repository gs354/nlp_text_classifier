from pathlib import Path
import pandas as pd
from nlp.validation import make_validation_set
from nlp.dataset import (
    make_batch_dataset,
    vectorize_text,
    preprocess_datasets,
)
from nlp.models import get_model, run_model, evaluate_model, make_predictions
from nlp.plotting import accuracy_loss_plot

# Data paths
DATA_DIR = "./data"
TRAIN_DATA = Path(f"{DATA_DIR}/imdb_data/train")
TEST_DATA = Path(f"{DATA_DIR}/imdb_data/test")
VALIDATION_DATA = Path(f"{DATA_DIR}/imdb_data/val")
TRUSTPILOT_REVIEWS = Path(
    f"{DATA_DIR}/web_reviews/trustpilot_reviews_octopus.energy?page=2_20231011100306.csv"
)

# Parameters
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 32
NGRAMS = 2
MAX_TOKENS = 20000
OUTPUT_MODE = "multi_hot"  # "tf_idf" #
HIDDEN_DIM = 16
OPTIMIZER = "rmsprop"
LOSS = "binary_crossentropy"
SAVENAME = Path(f"{DATA_DIR}/saved_models/binary_{NGRAMS}gram_{OUTPUT_MODE}.h5")
EPOCHS = 10
CACHE = True
PLOTSAVEDIR = Path(f"{DATA_DIR}/plots")


def main():
    make_validation_set(
        train_dir=TRAIN_DATA,
        val_dir=VALIDATION_DATA,
        val_fraction=VALIDATION_FRACTION,
        seed=1337,
    )

    train_ds = make_batch_dataset(TRAIN_DATA, BATCH_SIZE)
    val_ds = make_batch_dataset(VALIDATION_DATA, BATCH_SIZE)
    test_ds = make_batch_dataset(TEST_DATA, BATCH_SIZE)

    text_vectorization = vectorize_text(
        train_ds, ngrams=NGRAMS, max_tokens=MAX_TOKENS, output_mode=OUTPUT_MODE
    )

    train_ds = preprocess_datasets(train_ds, text_vectorization, num_parallel_calls=4)
    val_ds = preprocess_datasets(val_ds, text_vectorization, num_parallel_calls=4)
    test_ds = preprocess_datasets(test_ds, text_vectorization, num_parallel_calls=4)

    for inputs, targets in train_ds:
        print("inputs.shape:", inputs.shape)
        print("inputs.dtype:", inputs.dtype)
        print("targets.shape:", targets.shape)
        print("targets.dtype:", targets.dtype)
        print("inputs[0]:", inputs[0])
        print("targets[0]:", targets[0])
        print()
        break

    model = get_model(
        input_size=MAX_TOKENS,
        hidden_dim=HIDDEN_DIM,
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=["accuracy"],
    )

    history = run_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS,
        cache=CACHE,
        savename=SAVENAME,
    )

    accuracy_loss_plot(
        history=history,
        savedir=PLOTSAVEDIR,
    )

    evaluate_model(saved_model=SAVENAME, test_ds=test_ds)

    # Now fetch Trustpilot reviews and make predictions
    df = pd.read_csv(TRUSTPILOT_REVIEWS, index_col=0)
    reviews_list = df["review_text"].tolist()

    make_predictions(
        saved_model=SAVENAME,
        text_vectorization=text_vectorization,
        test_reviews=reviews_list,
    )


if __name__ == "__main__":
    main()
