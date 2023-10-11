import tensorflow as tf

layers = tf.keras.layers
Dataset = tf.data.Dataset
TextVectorization = tf.keras.layers.TextVectorization


def get_model(
    input_size: int,
    hidden_dim: int,
    optimizer: str,
    loss: str,
    metrics: list[str],
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_size,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def get_model_deep(
    input_size: int,
    embedding_dim: int,
    optimizer: str,
    loss: str,
    metrics: list[str],
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None,), dtype="int64")
    x = layers.Embedding(input_dim=input_size, output_dim=embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    outputs = layers.Dense(1, activation="sigmoid", name="outputs")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def run_model(
    model: tf.keras.Model,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int,
    cache: bool,
    savename: str,
):
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f"{savename}", save_best_only=True)]

    if cache:
        history = model.fit(
            train_ds.cache(),
            validation_data=val_ds.cache(),
            epochs=epochs,
            callbacks=callbacks,
        )

    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

    return history


def evaluate_model(saved_model: str, test_ds: Dataset) -> None:
    model = tf.keras.models.load_model(saved_model)
    print(f"Test acc: {model.evaluate(test_ds)[1]:.3f}")


def make_predictions(
    saved_model: str,
    text_vectorization: TextVectorization,
    test_reviews: list[list[str]],
) -> None:
    model = tf.keras.models.load_model(saved_model)
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    processed_inputs = text_vectorization(inputs)
    outputs = model(processed_inputs)
    inference_model = tf.keras.Model(inputs, outputs)

    raw_text_data = tf.convert_to_tensor(test_reviews)

    predictions = inference_model(raw_text_data)
    for i in range(len(predictions)):
        print(f"{raw_text_data[i]}: {float(predictions[i] * 100):.2f} percent positive")

    pass
