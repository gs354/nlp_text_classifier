from pathlib import Path
import tensorflow as tf
import string
import re

TextVectorization = tf.keras.layers.TextVectorization
Dataset = tf.data.Dataset


def make_batch_dataset(file_dir: Path, batch_size: int) -> Dataset:
    return tf.keras.utils.text_dataset_from_directory(file_dir, batch_size=batch_size)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


def vectorize_text(
    train_ds: Dataset, ngrams: int | None, max_tokens: int, output_mode: str
) -> TextVectorization:
    text_vectorization = TextVectorization(
        standardize=custom_standardization,
        ngrams=ngrams,
        max_tokens=max_tokens,
        output_mode=output_mode,
    )
    # Prepare a dataset that only yields raw text inputs (no labels):
    text_only_train_ds = train_ds.map(lambda x, y: x)
    # Use that dataset to index the dataset vocabulary via the adapt() method:
    text_vectorization.adapt(text_only_train_ds)

    return text_vectorization


def preprocess_datasets(
    dataset: Dataset, text_vectorization: TextVectorization, num_parallel_calls: int
) -> Dataset:
    return dataset.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=num_parallel_calls
    )
