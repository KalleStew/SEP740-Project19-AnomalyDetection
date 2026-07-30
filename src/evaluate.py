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


def _grid_shape(n_items: int, max_cols: int = 2) -> tuple[int, int]:
	"""Compute a near-square (rows, cols) grid layout for ``n_items`` subplots."""

	n_cols = min(max_cols, n_items) or 1
	n_rows = -(-n_items // n_cols)  # ceil division
	return n_rows, n_cols


def plot_reconstruction_error_distribution_comparison(
	model_errors: dict[str, dict[str, Any]],
	save_path: Optional[Path] = None,
) -> None:
	"""Plot reconstruction error distributions for multiple models in one figure.

	Args:
		model_errors: Mapping from model name to a dict with keys ``train_errors``,
			``test_errors``, ``y_test``, and ``threshold``.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt
	import seaborn as sns

	n_rows, n_cols = _grid_shape(len(model_errors))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
	axes_flat = axes.flatten()

	for ax, (model_name, data) in zip(axes_flat, model_errors.items()):
		train_errors = data["train_errors"]
		test_errors = data["test_errors"]
		y_test = data["y_test"]
		threshold = data["threshold"]

		sns.histplot(train_errors, bins=50, alpha=0.5, label="Train (Normal)", color="blue", stat="density", ax=ax)
		sns.histplot(test_errors[y_test == 0], bins=50, alpha=0.5, label="Test Normal", color="green", stat="density", ax=ax)
		sns.histplot(test_errors[y_test == 1], bins=50, alpha=0.5, label="Test Anomaly", color="red", stat="density", ax=ax)
		ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")
		ax.set_xlabel("Reconstruction Error (MSE)")
		ax.set_ylabel("Density")
		ax.set_title(str(model_name))
		ax.legend(fontsize=8)
		ax.grid(True, alpha=0.3)

	for ax in axes_flat[len(model_errors):]:
		ax.axis("off")

	fig.suptitle("Reconstruction Error Distribution by Model", fontsize=14)
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


def plot_confusion_matrix_comparison(
	cms_by_model: dict[str, np.ndarray],
	save_path: Optional[Path] = None,
) -> None:
	"""Plot confusion matrices for multiple models as a grid of heatmaps.

	Args:
		cms_by_model: Mapping from model name to a confusion matrix array.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt
	import seaborn as sns

	n_rows, n_cols = _grid_shape(len(cms_by_model))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
	axes_flat = axes.flatten()

	for ax, (model_name, cm) in zip(axes_flat, cms_by_model.items()):
		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
					xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], ax=ax)
		ax.set_xlabel("Predicted Label")
		ax.set_ylabel("True Label")
		ax.set_title(str(model_name))

	for ax in axes_flat[len(cms_by_model):]:
		ax.axis("off")

	fig.suptitle("Confusion Matrix by Model", fontsize=14)
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


def plot_precision_recall_curve_comparison(
	y_test: np.ndarray,
	model_errors: dict[str, dict[str, Any]],
	save_path: Optional[Path] = None,
) -> None:
	"""Overlay precision-recall curves for multiple models on a single plot.

	Args:
		y_test: Ground-truth test labels (shared across models).
		model_errors: Mapping from model name to a dict with keys ``test_errors``
			and ``threshold``.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt
	from sklearn.metrics import precision_recall_curve, auc

	plt.figure(figsize=(9, 7))
	colors = plt.get_cmap("tab10").colors

	for idx, (model_name, data) in enumerate(model_errors.items()):
		test_errors = data["test_errors"]
		threshold = data["threshold"]
		color = colors[idx % len(colors)]

		precision, recall, _ = precision_recall_curve(y_test, test_errors)
		pr_auc = auc(recall, precision)
		plt.plot(recall, precision, color=color, lw=2, label=f"{model_name} (AUC={pr_auc:.4f})")

		y_pred = predict_anomalies(test_errors, threshold)
		current_precision = precision_score(y_test, y_pred, zero_division=0)
		current_recall = recall_score(y_test, y_pred, zero_division=0)
		plt.scatter([current_recall], [current_precision], color=color, s=80, zorder=5, edgecolor="black")

	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall Curve Comparison")
	plt.legend(fontsize=9)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_reconstruction_error_vs_index_comparison(
	model_results: dict[str, dict[str, Any]],
	save_path: Optional[Path] = None,
) -> None:
	"""Plot reconstruction error vs sample index for multiple models in one figure.

	Args:
		model_results: Mapping from model name to a dict with keys ``test_errors``,
			``y_pred``, and ``threshold``.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt

	n_rows, n_cols = _grid_shape(len(model_results))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), squeeze=False)
	axes_flat = axes.flatten()

	for ax, (model_name, data) in zip(axes_flat, model_results.items()):
		test_errors = data["test_errors"]
		y_pred = data["y_pred"]
		threshold = data["threshold"]

		indices = np.arange(len(test_errors))
		colors = np.where(y_pred == 1, "red", "blue")
		ax.scatter(indices, test_errors, c=colors, alpha=0.6, s=8)
		ax.axhline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")
		ax.set_xlabel("Sample Index")
		ax.set_ylabel("Reconstruction Error (MSE)")
		ax.set_title(str(model_name))
		ax.legend(fontsize=8)
		ax.grid(True, alpha=0.3)

	for ax in axes_flat[len(model_results):]:
		ax.axis("off")

	fig.suptitle("Reconstruction Error vs Sample Index by Model", fontsize=14)
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


def plot_threshold_sensitivity_comparison(
	sensitivity_by_model: dict[str, Any],
	current_percentile: float,
	save_path: Optional[Path] = None,
) -> None:
	"""Overlay threshold-sensitivity curves for multiple models.

	Args:
		sensitivity_by_model: Mapping from model name to a DataFrame (or
			DataFrame-like object) with columns ``percentile``, ``threshold``,
			``accuracy``, ``precision``, ``recall``, and ``f1_score``.
		current_percentile: The percentile currently used for the operating threshold.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt

	colors = plt.get_cmap("tab10").colors
	fig, axes = plt.subplots(2, 2, figsize=(13, 9))

	for idx, (model_name, df) in enumerate(sensitivity_by_model.items()):
		color = colors[idx % len(colors)]
		axes[0, 0].plot(df["percentile"], df["threshold"], color=color, label=str(model_name))
		axes[0, 1].plot(df["percentile"], df["f1_score"], color=color, label=str(model_name))
		axes[1, 0].plot(df["percentile"], df["accuracy"], color=color, label=str(model_name))
		axes[1, 1].plot(df["recall"], df["precision"], color=color, label=str(model_name))

	axes[0, 0].axvline(current_percentile, color="k", linestyle="--", alpha=0.6)
	axes[0, 0].set_xlabel("Percentile")
	axes[0, 0].set_ylabel("Threshold")
	axes[0, 0].set_title("Threshold vs Percentile")
	axes[0, 0].grid(True, alpha=0.3)
	axes[0, 0].legend(fontsize=8)

	axes[0, 1].axvline(current_percentile, color="k", linestyle="--", alpha=0.6, label=f"Current ({current_percentile})")
	axes[0, 1].set_xlabel("Percentile")
	axes[0, 1].set_ylabel("F1-Score")
	axes[0, 1].set_title("F1-Score vs Percentile")
	axes[0, 1].grid(True, alpha=0.3)
	axes[0, 1].legend(fontsize=8)

	axes[1, 0].axvline(current_percentile, color="k", linestyle="--", alpha=0.6)
	axes[1, 0].set_xlabel("Percentile")
	axes[1, 0].set_ylabel("Accuracy")
	axes[1, 0].set_title("Accuracy vs Percentile")
	axes[1, 0].grid(True, alpha=0.3)
	axes[1, 0].legend(fontsize=8)

	axes[1, 1].set_xlabel("Recall")
	axes[1, 1].set_ylabel("Precision")
	axes[1, 1].set_title("Precision-Recall Tradeoff")
	axes[1, 1].grid(True, alpha=0.3)
	axes[1, 1].legend(fontsize=8)

	fig.suptitle("Threshold Sensitivity Comparison", fontsize=14)
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


def plot_threshold_sensitivity(
	sensitivity_df: Any,
	current_percentile: float,
	save_path: Optional[Path] = None,
) -> None:
	"""Plot how metrics change across threshold percentiles for a single model.

	Args:
		sensitivity_df: DataFrame (or DataFrame-like object) with columns
			``percentile``, ``threshold``, ``accuracy``, ``precision``, ``recall``,
			and ``f1_score``, as produced by sweeping threshold percentiles.
		current_percentile: The percentile currently used for the operating threshold.
		save_path: Optional path to save the figure as PNG.
	"""

	import matplotlib.pyplot as plt

	df = sensitivity_df
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	axes[0, 0].plot(df["percentile"], df["threshold"], "b-")
	axes[0, 0].set_xlabel("Percentile")
	axes[0, 0].set_ylabel("Threshold")
	axes[0, 0].set_title("Threshold vs Percentile")
	axes[0, 0].grid(True, alpha=0.3)

	axes[0, 1].plot(df["percentile"], df["precision"], "g-", label="Precision")
	axes[0, 1].plot(df["percentile"], df["recall"], "r-", label="Recall")
	axes[0, 1].plot(df["percentile"], df["f1_score"], "b-", label="F1-Score")
	axes[0, 1].axvline(current_percentile, color="k", linestyle="--", label=f"Current ({current_percentile})")
	axes[0, 1].set_xlabel("Percentile")
	axes[0, 1].set_ylabel("Score")
	axes[0, 1].set_title("Precision, Recall, F1 vs Percentile")
	axes[0, 1].legend()
	axes[0, 1].grid(True, alpha=0.3)

	axes[1, 0].plot(df["percentile"], df["accuracy"], "m-")
	axes[1, 0].axvline(current_percentile, color="k", linestyle="--")
	axes[1, 0].set_xlabel("Percentile")
	axes[1, 0].set_ylabel("Accuracy")
	axes[1, 0].set_title("Accuracy vs Percentile")
	axes[1, 0].grid(True, alpha=0.3)

	axes[1, 1].plot(df["recall"], df["precision"], "c-")
	current_idx = int(np.argmin(np.abs(df["percentile"] - current_percentile)))
	axes[1, 1].scatter(
		[df["recall"].iloc[current_idx]], [df["precision"].iloc[current_idx]],
		color="red", s=100, zorder=5, label=f"Current ({current_percentile})"
	)
	axes[1, 1].set_xlabel("Recall")
	axes[1, 1].set_ylabel("Precision")
	axes[1, 1].set_title("Precision-Recall Tradeoff")
	axes[1, 1].legend()
	axes[1, 1].grid(True, alpha=0.3)

	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
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


def build_evaluation_report(
	metrics: dict[str, Any],
	threshold: float,
	percentile: float,
) -> str:
	"""Build the comprehensive evaluation report as a string.

	Args:
		metrics: Dictionary from compute_anomaly_metrics.
		threshold: Anomaly threshold value.
		percentile: Percentile used for threshold selection.

	Returns:
		The formatted report text.
	"""

	lines = []
	lines.append("AUTOENCODER ANOMALY DETECTION - EVALUATION REPORT")
	lines.append("=" * 60)
	lines.append("")
	lines.append(f"Threshold Percentile: {percentile}")
	lines.append(f"Anomaly Threshold:    {threshold:.8f}")
	lines.append("")
	lines.append("CLASSIFICATION METRICS")
	lines.append("-" * 30)
	lines.append(f"Accuracy:  {metrics['accuracy']:.6f}")
	lines.append(f"Precision: {metrics['precision']:.6f}")
	lines.append(f"Recall:    {metrics['recall']:.6f}")
	lines.append(f"F1-Score:  {metrics['f1_score']:.6f}")
	lines.append("")
	lines.append("CLASSIFICATION REPORT")
	lines.append("-" * 30)
	lines.append(str(metrics["classification_report"]))
	lines.append("")
	lines.append("CONFUSION MATRIX")
	lines.append("-" * 30)
	lines.append(f"{metrics['confusion_matrix']}")
	lines.append("")
	lines.append("=" * 60)
	lines.append("End of Report")
	lines.append("=" * 60)
	return "\n".join(lines) + "\n"


def save_evaluation_report(
	metrics: dict[str, Any],
	threshold: float,
	percentile: float,
	save_path: Optional[Path] = None,
) -> str:
	"""Build the comprehensive evaluation report, optionally saving it to disk.

	Args:
		metrics: Dictionary from compute_anomaly_metrics.
		threshold: Anomaly threshold value.
		percentile: Percentile used for threshold selection.
		save_path: Optional path to save the report as a text file. When omitted,
			the report is only returned (and can be printed/displayed inline).

	Returns:
		The formatted report text.
	"""

	report = build_evaluation_report(metrics, threshold, percentile)

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		with save_path.open("w", encoding="utf-8") as f:
			f.write(report)

	return report
