"""Training utilities for the autoencoder anomaly detection pipeline.

Methodology
-----------
The autoencoder is trained in an unsupervised fashion, using only *normal*
network traffic as both the model input and the reconstruction target
(``model.fit(x=X_train, y=X_train, ...)``). No attack traffic is used during
training. This is the defining characteristic of the reconstruction-based
anomaly-detection approach used throughout this project: because the model
never sees attacks during training, it learns a latent representation that is
specialized to normal traffic and therefore reconstructs attacks poorly. The
resulting per-sample reconstruction error is later converted into a binary
anomaly decision in ``src/evaluate.py``.

Training minimizes mean squared reconstruction error using the Adam optimizer
(Kingma & Ba, 2014, "Adam: A Method for Stochastic Optimization"), a standard
choice for training deep neural networks due to its adaptive per-parameter
learning rates.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf

try:
	from .model import Autoencoder, build_autoencoder
except ImportError:  # pragma: no cover - fallback for notebook sys.path imports
	from model import Autoencoder, build_autoencoder


def set_global_seed(seed: int = 42) -> None:
	"""Set Python, NumPy, and TensorFlow seeds for reproducibility.

	Args:
		seed: Seed value applied across the supported random number generators.
	"""

	seed = int(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def get_device_name() -> str:
	"""Select the best available device in a hardware-agnostic order.

	Returns:
		A TensorFlow device name string for CUDA, Apple Silicon MPS, or CPU.
	"""

	physical_gpus = tf.config.list_physical_devices("GPU")
	if physical_gpus:
		return "/GPU:0"

	physical_mps = tf.config.list_physical_devices("MPS")
	if physical_mps:
		return "/MPS:0"

	return "/CPU:0"


def create_autoencoder(
	input_dim: int,
	latent_dim: int = 16,
	hidden_units: tuple[int, ...] = (32,),
	dropout_rate: float = 0.0,
	n_hidden_layers: int = 1,
	activation_encoder: str = "relu",
	activation_decoder: str = "relu",
	output_activation: str = "linear",
) -> tf.keras.Model:
	"""Create the project autoencoder model.

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
		A TensorFlow autoencoder model.
	"""

	return build_autoencoder(
		input_dim=input_dim,
		latent_dim=latent_dim,
		hidden_units=hidden_units,
		dropout_rate=dropout_rate,
		n_hidden_layers=n_hidden_layers,
		activation_encoder=activation_encoder,
		activation_decoder=activation_decoder,
		output_activation=output_activation,
	)


def compile_autoencoder(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
	"""Compile an autoencoder for reconstruction loss minimization.

	Args:
		model: Uncompiled TensorFlow model.
		learning_rate: Adam learning rate.

	Returns:
		The compiled model.
	"""

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	# Mean squared error (MSE) is the standard reconstruction loss for
	# continuous, standardized features: it directly measures how far the
	# reconstructed sample is from the original, which is exactly the
	# quantity later used as the anomaly score.
	model.compile(optimizer=optimizer, loss="mse")
	return model


def train_autoencoder(
	X_train: np.ndarray,
	input_dim: int,
	latent_dim: int = 16,
	hidden_units: tuple[int, ...] = (32,),
	activation: str = "relu",
	n_hidden_layers: int = 1,
	output_activation: str = "linear",
	learning_rate: float = 1e-3,
	epochs: int = 50,
	batch_size: int = 256,
	validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
	seed: int = 42,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
	"""Train the autoencoder on the supplied training data.

	Args:
		X_train: Training features used as both input and reconstruction target.
		input_dim: Number of input features.
		latent_dim: Size of the bottleneck representation.
		hidden_units: Hidden-layer widths used symmetrically in encoder and decoder.
		activation: Activation function used in hidden layers.
		output_activation: Activation function used in the reconstruction layer.
		learning_rate: Adam learning rate.
		epochs: Number of training epochs.
		batch_size: Training batch size.
		validation_data: Optional validation tuple ``(X_val, y_val)`` or ``(X_val, X_val)``.
		seed: Seed used for deterministic initialization.

	Returns:
		A tuple containing the trained model and the Keras History object.
	"""

	set_global_seed(seed)
	device_name = get_device_name()

	model = create_autoencoder(
		input_dim=input_dim,
		latent_dim=latent_dim,
		hidden_units=hidden_units,
		dropout_rate=0.0,
		n_hidden_layers=n_hidden_layers,
		activation_encoder=activation,
		activation_decoder=activation,
		output_activation=output_activation,
	)
	model = compile_autoencoder(model, learning_rate=learning_rate)

	# The autoencoder is self-supervised: the training target ("y") is the
	# input itself ("x"), so the model is optimized purely to reconstruct
	# normal traffic. No labels are used at any point during training.
	fit_kwargs: dict[str, Any] = {
		"x": X_train,
		"y": X_train,
		"epochs": epochs,
		"batch_size": batch_size,
		"verbose": 1,
	}
	if validation_data is not None:
		fit_kwargs["validation_data"] = validation_data

	with tf.device(device_name):
		history = model.fit(**fit_kwargs)

	return model, history


def save_model(model: tf.keras.Model, model_path: Path) -> Path:
	"""Save a trained model to disk.

	Args:
		model: Trained TensorFlow model.
		model_path: Destination path for the saved model.

	Returns:
		The resolved path to the saved model.
	"""

	model_path = Path(model_path)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	model.save(model_path)
	return model_path
