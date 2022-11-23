import logging
from typing import Optional
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from src.models.mpnn import MPNNModel

LOGGER = logging.getLogger(__name__)


class MPNNTrainer:
    """
    Trainer for MPNN
    """
    def __init__(
        self,
        x_train: tuple,
        batch_size: int = 32,
        message_units: int = 64,
        message_steps: int = 4,
        num_attention_heads: int = 8,
        dense_units: int = 512,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=5e-4),
        class_weight: dict = None,
    ) -> None:
        self.node_dim = x_train[0][0][0].shape[0]
        self.edge_dim = x_train[1][0][0].shape[0]
        self.batch_size = batch_size
        self.message_units = message_units
        self.message_steps = message_steps
        self.num_attention_heads = num_attention_heads
        self.dense_units = dense_units
        self.model = self.setup_model()
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.compile_model()
        self.plot_model()

    def setup_model(self) -> MPNNModel:
        """
        Creates GNN model based on the data type (weighted/unweighted).
        """
        model = MPNNModel(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            batch_size=self.message_units,
            message_units=self.message_units,
            num_attention_heads=self.num_attention_heads,
            dense_units=self.dense_units
        )
        return model

    def compile_model(self):
        self.model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=self.optimizer,
            metrics=[keras.metrics.AUC(name="AUC")],
        )

    def plot_model(self):
        keras.utils.plot_model(self.model, show_dtype=True, show_shapes=True)

    def train(
            self,
            train_dataset: tf.data.Dataset,
            valid_dataset: tf.data.Dataset,
            epochs: int = 40
    ) -> keras.callbacks.History:

        history = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            verbose=2,
            class_weight=self.class_weight,
        )

        plt.figure(figsize=(10, 6))
        plt.plot(history.history["AUC"], label="train AUC")
        plt.plot(history.history["val_AUC"], label="valid AUC")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("AUC", fontsize=16)
        plt.legend(fontsize=16)
        return history

    def predict(self, dataset):
        return tf.squeeze(self.model.predict(dataset), axis=1)

    def save(self, path: Optional[str] = "models/mpnn_model.pt"):
        """
        Saves trained model.
        :param path: (str, optional) path with model name to save the trained model.
        """
        self.model.save(path)

    def load(self, path: Optional[str] = "models/mpnn_model.pt"):
        """
        Loads pretrained model.
        :param path: (str, optional): path of pretrained model.
        """
        self.model = tf.keras.models.load_model(path)
