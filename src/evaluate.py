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


def plot_reconstruction_error_distribution(
	train_errors: np.ndarray,
	test_errors: np.ndarray,
	y_test: np.ndarray,
	threshold: float,
	save_path: Optional[Path] = None,
) -> None:
	"""Plot histogram of reconstruction errors for train and test sets.

	Args:
		train_errors: Reconstruction errors from training data.
		test_errors: Reconstruction errors from test data.
		y_test: Ground-truth test labels (0=normal, 1=anomaly).
		threshold: Anomaly threshold value.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt
	import seaborn as sns

	plt.figure(figsize=(10, 6))
	sns.histplot(train_errors, bins=50, alpha=0.5, label="Train (Normal)", color="blue", stat="density")
	sns.histplot(test_errors[y_test == 0], bins=50, alpha=0.5, label="Test Normal", color="green", stat="density")
	sns.histplot(test_errors[y_test == 1], bins=50, alpha=0.5, label="Test Anomaly", color="red", stat="density")
	plt.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")
	plt.xlabel("Reconstruction Error (MSE)")
	plt.ylabel("Density")
	plt.title("Reconstruction Error Distribution")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_confusion_matrix(
	cm: np.ndarray,
	save_path: Optional[Path] = None,
) -> None:
	"""Plot confusion matrix as a heatmap.

	Args:
		cm: Confusion matrix array from sklearn.metrics.confusion_matrix.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt
	import seaborn as sns

	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
				xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")
	plt.title("Confusion Matrix")
	plt.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_precision_recall_curve(
	y_test: np.ndarray,
	test_errors: np.ndarray,
	threshold: float,
	save_path: Optional[Path] = None,
) -> None:
	"""Plot precision-recall curve using reconstruction errors as scores.

	Args:
		y_test: Ground-truth test labels.
		test_errors: Reconstruction errors for test samples.
		threshold: Current anomaly threshold (marked on curve).
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt
	from sklearn.metrics import precision_recall_curve, auc

	precision, recall, thresholds = precision_recall_curve(y_test, test_errors)
	pr_auc = auc(recall, precision)

	plt.figure(figsize=(8, 6))
	plt.plot(recall, precision, color="blue", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")

	# Mark current threshold point
	y_pred = predict_anomalies(test_errors, threshold)
	current_precision = precision_score(y_test, y_pred, zero_division=0)
	current_recall = recall_score(y_test, y_pred, zero_division=0)
	plt.scatter([current_recall], [current_precision], color="red", s=100, zorder=5,
				label=f"Current Threshold (P={current_precision:.3f}, R={current_recall:.3f})")

	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall Curve")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_reconstruction_error_vs_index(
	test_errors: np.ndarray,
	y_pred: np.ndarray,
	threshold: float,
	save_path: Optional[Path] = None,
) -> None:
	"""Plot reconstruction error vs sample index with anomaly highlights.

	Args:
		test_errors: Reconstruction errors for test samples.
		y_pred: Binary anomaly predictions.
		threshold: Anomaly threshold value.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt

	plt.figure(figsize=(12, 5))
	indices = np.arange(len(test_errors))
	colors = np.where(y_pred == 1, "red", "blue")
	plt.scatter(indices, test_errors, c=colors, alpha=0.6, s=10, label="Samples")
	plt.axhline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")
	plt.xlabel("Sample Index")
	plt.ylabel("Reconstruction Error (MSE)")
	plt.title("Reconstruction Error vs Sample Index")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def print_error_statistics(
	train_errors: np.ndarray,
	test_errors: np.ndarray,
	y_test: np.ndarray,
	threshold: float,
) -> None:
	"""Print summary statistics of reconstruction errors.

	Args:
		train_errors: Reconstruction errors from training data.
		test_errors: Reconstruction errors from test data.
		y_test: Ground-truth test labels.
		threshold: Anomaly threshold value.
	"""

	print("=" * 60)
	print("RECONSTRUCTION ERROR STATISTICS")
	print("=" * 60)
	print(f"\nTrain Errors (Normal):")
	print(f"  Count: {len(train_errors)}")
	print(f"  Mean:  {np.mean(train_errors):.6f}")
	print(f"  Std:   {np.std(train_errors):.6f}")
	print(f"  Min:   {np.min(train_errors):.6f}")
	print(f"  Max:   {np.max(train_errors):.6f}")
	print(f"  Median:{np.median(train_errors):.6f}")

	print(f"\nTest Errors (Normal):")
	test_normal = test_errors[y_test == 0]
	print(f"  Count: {len(test_normal)}")
	print(f"  Mean:  {np.mean(test_normal):.6f}")
	print(f"  Std:   {np.std(test_normal):.6f}")
	print(f"  Min:   {np.min(test_normal):.6f}")
	print(f"  Max:   {np.max(test_normal):.6f}")
	print(f"  Median:{np.median(test_normal):.6f}")

	print(f"\nTest Errors (Anomaly):")
	test_anomaly = test_errors[y_test == 1]
	print(f"  Count: {len(test_anomaly)}")
	print(f"  Mean:  {np.mean(test_anomaly):.6f}")
	print(f"  Std:   {np.std(test_anomaly):.6f}")
	print(f"  Min:   {np.min(test_anomaly):.6f}")
	print(f"  Max:   {np.max(test_anomaly):.6f}")
	print(f"  Median:{np.median(test_anomaly):.6f}")

	print(f"\nThreshold: {threshold:.6f}")
	print(f"  Train samples above threshold: {np.sum(train_errors > threshold)} ({100 * np.mean(train_errors > threshold):.2f}%)")
	print(f"  Test normal above threshold:   {np.sum(test_normal > threshold)} ({100 * np.mean(test_normal > threshold):.2f}%)")
	print(f"  Test anomaly above threshold:  {np.sum(test_anomaly > threshold)} ({100 * np.mean(test_anomaly > threshold):.2f}%)")
	print("=" * 60)


def save_evaluation_report(
	metrics: dict[str, Any],
	threshold: float,
	percentile: float,
	save_path: Path,
) -> None:
	"""Save comprehensive evaluation report to a text file.

	Args:
		metrics: Dictionary from compute_anomaly_metrics.
		threshold: Anomaly threshold value.
		percentile: Percentile used for threshold selection.
		save_path: Path to save the report.
	"""

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	with save_path.open("w", encoding="utf-8") as f:
		f.write("AUTOENCODER ANOMALY DETECTION - EVALUATION REPORT\n")
		f.write("=" * 60 + "\n\n")
		f.write(f"Threshold Percentile: {percentile}\n")
		f.write(f"Anomaly Threshold:    {threshold:.8f}\n\n")
		f.write("CLASSIFICATION METRICS\n")
		f.write("-" * 30 + "\n")
		f.write(f"Accuracy:  {metrics['accuracy']:.6f}\n")
		f.write(f"Precision: {metrics['precision']:.6f}\n")
		f.write(f"Recall:    {metrics['recall']:.6f}\n")
		f.write(f"F1-Score:  {metrics['f1_score']:.6f}\n\n")
		f.write("CLASSIFICATION REPORT\n")
		f.write("-" * 30 + "\n")
		f.write(str(metrics["classification_report"]))
		f.write("\n\n")
		f.write("CONFUSION MATRIX\n")
		f.write("-" * 30 + "\n")
		f.write(f"{metrics['confusion_matrix']}\n")
		f.write("\n")
		f.write("=" * 60 + "\n")
		f.write("End of Report\n")
		f.write("=" * 60 + "\n")
