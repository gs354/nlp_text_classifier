import matplotlib.pyplot as plt


def accuracy_loss_plot(history, savedir: str) -> None:
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(accuracy) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, accuracy, "bo", label="Training accuracy")
    ax[0].plot(epochs, val_accuracy, "r", label="Validation accuracy")
    ax[0].set_title("Training and validation accuracy")
    ax[0].legend()

    ax[1].plot(epochs, loss, "bo", label="Training loss")
    ax[1].plot(epochs, val_loss, "r", label="Validation loss")
    ax[1].set_title("Training and validation loss")
    ax[1].legend()
    plt.savefig(f"{savedir}/_accuracy_loss.pdf")
