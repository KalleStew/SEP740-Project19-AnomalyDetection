"""Model definitions for the anomaly detection autoencoder pipeline.

Architecture and methodology references
----------------------------------------
The autoencoder architecture implemented here (a symmetric, fully connected
encoder-decoder pair with a low-dimensional latent bottleneck) follows the
standard deep-autoencoder design introduced by Hinton & Salakhutdinov (2006),
"Reducing the Dimensionality of Data with Neural Networks", Science, 313(5786).

Using reconstruction error from such an autoencoder as an anomaly score is a
well-established technique in the anomaly-detection literature (e.g.,
Sakurada & Yairi, 2014, "Anomaly Detection Using Autoencoders with Nonlinear
Dimensionality Reduction"). The model is trained exclusively on normal
traffic, so it learns to reconstruct normal patterns accurately; samples that
the trained model reconstructs poorly (i.e., attacks it has never seen) are
flagged as anomalous. The actual thresholding logic that turns reconstruction
error into a binary anomaly decision lives in ``src/evaluate.py``.

The model is built with the Keras functional/subclassing API on top of
TensorFlow (Abadi et al., 2016; Chollet, 2015).
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Sequential, layers

DEFAULT_HIDDEN_UNITS: tuple[int, ...] = (32,)


def build_autoencoder(
    input_dim: int,
    latent_dim: int = 16,
    hidden_units: Sequence[int] | int = DEFAULT_HIDDEN_UNITS,
    dropout_rate: float = 0.0,
    n_hidden_layers: int = 1,
    activation_encoder: str = "relu",
    activation_decoder: str = "relu",
    output_activation: str = "linear",
) -> tf.keras.Model:
    """Build a compact fully connected autoencoder.

    Args:
        input_dim: Number of input features.
        latent_dim: Size of the bottleneck representation.
        hidden_units: Hidden-layer widths used symmetrically in encoder and decoder.
        dropout_rate: Optional dropout rate applied after hidden layers.
        n_hidden_layers: Number of hidden layers per encoder/decoder side.
        activation_encoder: Activation function used in encoder layers.
        activation_decoder: Activation function used in decoder layers.
        output_activation: Activation function used in the reconstruction layer.

    Returns:
        A TensorFlow autoencoder model instance.
    """

    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)

    input_dim = int(input_dim)
    latent_dim = int(latent_dim)
    hidden_units = tuple(int(unit) for unit in hidden_units)
    dropout_rate = float(dropout_rate)
    n_hidden_layers = int(n_hidden_layers)

    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")
    if not hidden_units:
        raise ValueError("hidden_units must contain at least one layer width.")
    if n_hidden_layers <= 0:
        raise ValueError("n_hidden_layers must be a positive integer.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    # Encoder: progressively compresses the input down to `latent_dim`. This
    # bottleneck forces the network to learn a compact representation that
    # captures the dominant patterns in normal traffic.
    encoder = Sequential(name="encoder")
    encoder.add(layers.InputLayer(shape=(input_dim,)))
    for _ in range(n_hidden_layers):
        for units in hidden_units:
            encoder.add(layers.Dense(units, activation=activation_encoder))
            if dropout_rate > 0:
                encoder.add(layers.Dropout(dropout_rate))
    encoder.add(layers.Dense(latent_dim, activation=activation_encoder))

    # Decoder: mirrors the encoder to reconstruct the original input from the
    # latent representation. Reconstruction quality on unseen data is what
    # drives the anomaly score at evaluation time (see src/evaluate.py).
    decoder = Sequential(name="decoder")
    for _ in range(n_hidden_layers):
        for units in reversed(hidden_units):
            decoder.add(layers.Dense(units, activation=activation_decoder))
            if dropout_rate > 0:
                decoder.add(layers.Dropout(dropout_rate))
    decoder.add(layers.Dense(input_dim, activation=output_activation))

    inputs = layers.Input(shape=(input_dim,), name="input_features")
    outputs = decoder(encoder(inputs))
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")


class Autoencoder(tf.keras.Model):
    """Notebook-compatible autoencoder wrapper with stable serialization support."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_units: Sequence[int] | int = DEFAULT_HIDDEN_UNITS,
        dropout_rate: float = 0.0,
        n_hidden_layers: int = 1,
        activation_encoder: str = "relu",
        activation_decoder: str = "relu",
        output_activation: str = "linear",
    ) -> None:
        """Create an autoencoder with mirrored encoder/decoder stacks."""

        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = (hidden_units,)

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_units = tuple(int(unit) for unit in hidden_units)
        self.dropout_rate = float(dropout_rate)
        self.n_hidden_layers = int(n_hidden_layers)
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.output_activation = output_activation

        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not self.hidden_units:
            raise ValueError("hidden_units must contain at least one layer width.")
        if self.n_hidden_layers <= 0:
            raise ValueError("n_hidden_layers must be a positive integer.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")

        self.encoder = Sequential(name="encoder")
        self.encoder.add(layers.InputLayer(shape=(self.input_dim,)))
        for _ in range(self.n_hidden_layers):
            for units in self.hidden_units:
                self.encoder.add(layers.Dense(units, activation=self.activation_encoder))
                if self.dropout_rate > 0:
                    self.encoder.add(layers.Dropout(self.dropout_rate))
        self.encoder.add(layers.Dense(self.latent_dim, activation=self.activation_encoder))

        self.decoder = Sequential(name="decoder")
        for _ in range(self.n_hidden_layers):
            for units in reversed(self.hidden_units):
                self.decoder.add(layers.Dense(units, activation=self.activation_decoder))
                if self.dropout_rate > 0:
                    self.decoder.add(layers.Dropout(self.dropout_rate))
        self.decoder.add(layers.Dense(self.input_dim, activation=self.output_activation))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Run a forward pass through the encoder and decoder."""

        return self.decoder(self.encoder(inputs))

    def get_config(self) -> dict:
        """Return the configuration required to serialize the model."""

        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "n_hidden_layers": self.n_hidden_layers,
                "activation_encoder": self.activation_encoder,
                "activation_decoder": self.activation_decoder,
                "output_activation": self.output_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "Autoencoder":
        """Recreate the model from a serialized configuration."""

        return cls(**config)
