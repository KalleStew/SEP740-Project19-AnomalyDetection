"""Evaluation helpers for the autoencoder-based anomaly detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def calculate_reconstruction_error(model: Any, data: np.ndarray) -> np.ndarray:
	"""Compute per-sample mean squared reconstruction error.

	Args:
		model: Trained autoencoder with a ``predict`` method.
		data: Input array used for reconstruction.

	Returns:
		A one-dimensional array containing reconstruction error per sample.
	"""

	reconstructed = model.predict(data, verbose=0)
	return np.mean(np.square(data - reconstructed), axis=1)


def select_anomaly_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
	"""Select an anomaly threshold from a reference error distribution.

	Args:
		errors: Reconstruction errors from the calibration set.
		percentile: Percentile used to define the threshold.

	Returns:
		The threshold value as a float.
	"""

	if errors.size == 0:
		raise ValueError("errors must contain at least one value.")
	if not 0 < percentile < 100:
		raise ValueError("percentile must be between 0 and 100.")
	return float(np.percentile(errors, percentile))


def predict_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
	"""Convert reconstruction errors into binary anomaly predictions.

	Args:
		errors: Reconstruction errors for the evaluated samples.
		threshold: Decision threshold above which a sample is flagged as anomalous.

	Returns:
		Binary anomaly predictions where ``1`` means anomaly and ``0`` means normal.
	"""

	return np.where(errors > threshold, 1, 0)


def compute_anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
	"""Compute standard classification metrics for anomaly detection.

	Args:
		y_true: Ground-truth labels.
		y_pred: Predicted binary anomaly labels.

	Returns:
		A dictionary containing accuracy, precision, recall, F1, report, and confusion matrix.
	"""

	metrics = {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
		"f1_score": f1_score(y_true, y_pred, zero_division=0),
		"classification_report": classification_report(y_true, y_pred, zero_division=0),
		"confusion_matrix": confusion_matrix(y_true, y_pred),
	}
	return metrics


def generate_anomaly_metrics_and_threshold(
	model: Any,
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	percentile: float = 95.0,
	metrics_path: Optional[Path] = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
	"""Calibrate a threshold and compute evaluation metrics for anomaly detection.

	Args:
		model: Trained autoencoder with a ``predict`` method.
		X_train: Training data used to calibrate the threshold.
		X_test: Test data used for evaluation.
		y_test: Ground-truth test labels.
		percentile: Percentile used to select the anomaly threshold.
		metrics_path: Optional text file path where a summary report should be written.

	Returns:
		A tuple containing the threshold, the binary predictions, and a metrics dictionary.
	"""

	train_errors = calculate_reconstruction_error(model, X_train)
	test_errors = calculate_reconstruction_error(model, X_test)
	threshold = select_anomaly_threshold(train_errors, percentile=percentile)
	y_pred = predict_anomalies(test_errors, threshold)
	metrics = compute_anomaly_metrics(y_test, y_pred)

	if metrics_path is not None:
		metrics_path = Path(metrics_path)
		metrics_path.parent.mkdir(parents=True, exist_ok=True)
		with metrics_path.open("w", encoding="utf-8") as handle:
			handle.write("Autoencoder Anomaly Detection Evaluation\n")
			handle.write("--------------------------------------\n")
			handle.write(f"Threshold percentile: {percentile}\n")
			handle.write(f"Anomaly threshold: {threshold:.8f}\n\n")
			handle.write(f"Accuracy:  {metrics['accuracy']:.6f}\n")
			handle.write(f"Precision: {metrics['precision']:.6f}\n")
			handle.write(f"Recall:    {metrics['recall']:.6f}\n")
			handle.write(f"F1-score:  {metrics['f1_score']:.6f}\n\n")
			handle.write("Classification report:\n")
			handle.write(str(metrics["classification_report"]))
			handle.write("\nConfusion matrix:\n")
			handle.write(f"{metrics['confusion_matrix']}\n")

	return threshold, y_pred, metrics
