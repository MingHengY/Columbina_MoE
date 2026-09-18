"""Validation-only temperature scaling and calibration metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def expected_calibration_error(labels, probabilities, bins=15):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        error += mask.sum() / total * abs(
            probabilities[mask].mean() - labels[mask].mean()
        )
    return float(error)


def calibration_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8
    )
    nll = -np.mean(
        labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
    )
    return {
        'nll': float(nll),
        'brier_score': float(np.mean(np.square(probabilities - labels))),
        'ece': expected_calibration_error(labels, probabilities),
    }


def _best_f1_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    candidates = np.unique(np.concatenate((
        np.asarray([0.0, 0.5, 1.0]), probabilities,
    )))
    best = (float('-inf'), 0.5)
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = int(((labels == 1) & predicted).sum())
        fp = int(((labels == 0) & predicted).sum())
        fn = int(((labels == 1) & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) \
            if precision + recall else 0.0
        candidate = (f1, -float(threshold))
        if candidate > (best[0], -best[1]):
            best = (f1, float(threshold))
    return best[1], best[0]


def fit_temperature_scaling(logits, labels):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(logits) != len(labels) or len(labels) < 2:
        raise ValueError('Calibration requires aligned logits and at least two labels')
    if len(np.unique(labels)) < 2:
        raise ValueError('Calibration requires both positive and negative validation labels')

    temperatures = np.exp(np.linspace(np.log(0.05), np.log(10.0), 600))
    losses = []
    for temperature in temperatures:
        scaled = logits / temperature
        losses.append(float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled)))
    temperature = float(temperatures[int(np.argmin(losses))])
    raw_probabilities = sigmoid(logits)
    calibrated_probabilities = sigmoid(logits / temperature)
    threshold, threshold_f1 = _best_f1_threshold(labels, calibrated_probabilities)
    residual_quantile = float(np.quantile(
        np.abs(labels - calibrated_probabilities), 0.90
    ))

    return {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'method': 'temperature_scaling_validation_only',
        'temperature': temperature,
        'validation_samples': int(len(labels)),
        'decision_threshold': float(threshold),
        'decision_threshold_source': 'validation_f1_after_temperature_scaling',
        'decision_threshold_f1': float(threshold_f1),
        'residual_quantile_90': residual_quantile,
        'raw': calibration_metrics(labels, raw_probabilities),
        'calibrated': calibration_metrics(labels, calibrated_probabilities),
    }


def apply_temperature(logits, calibration):
    temperature = float((calibration or {}).get('temperature', 1.0))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f'Invalid calibration temperature: {temperature}')
    return sigmoid(np.asarray(logits, dtype=np.float64) / temperature)


def save_calibration(output_dir, calibration):
    path = Path(output_dir) / 'calibration.json'
    path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


def load_calibration(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    temperature = float(payload.get('temperature', 0.0))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError('Calibration file contains an invalid temperature')
    return payload
