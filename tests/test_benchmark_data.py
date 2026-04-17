"""
Tests for the canonical benchmark data module.

``benchmark_data.py`` is the single source of truth for every number shown
in the README, the Gradio demo, and the training-metadata JSON. These
tests lock down its schema so we never ship inconsistent or malformed
metrics (which would be extremely embarrassing on a portfolio site).
"""

from __future__ import annotations

import pytest

from benchmark_data import (
    BENCHMARK_METRICS,
    CONFUSION_PAIRS,
    CONVERGENCE_HISTORY,
    TRAINING_CONFIG,
    accuracy_delta,
    available_models,
    best_model_key,
)

REQUIRED_KEYS = {
    "display_name",
    "test_accuracy",
    "trainable_params",
    "total_params",
    "size_mb",
    "latency_ms",
    "fps",
    "input_size",
    "normalization",
    "strategy",
    "available",
    "color",
}


def test_benchmark_metrics_is_non_empty() -> None:
    assert len(BENCHMARK_METRICS) >= 2


@pytest.mark.parametrize("model_name", list(BENCHMARK_METRICS.keys()))
def test_benchmark_row_has_all_required_keys(model_name: str) -> None:
    row = BENCHMARK_METRICS[model_name]
    missing = REQUIRED_KEYS - set(row.keys())
    assert not missing, f"{model_name} is missing keys: {missing}"


@pytest.mark.parametrize("model_name", list(BENCHMARK_METRICS.keys()))
def test_benchmark_accuracy_is_in_valid_range(model_name: str) -> None:
    """Test accuracies are stored as percentages in [0, 100]."""
    acc = BENCHMARK_METRICS[model_name]["test_accuracy"]
    assert 0.0 <= acc <= 100.0, f"{model_name}: accuracy {acc} out of range"


@pytest.mark.parametrize("model_name", list(BENCHMARK_METRICS.keys()))
def test_benchmark_numeric_fields_are_positive(model_name: str) -> None:
    row = BENCHMARK_METRICS[model_name]
    for field in ("trainable_params", "total_params", "size_mb", "latency_ms", "fps", "input_size"):
        value = row[field]
        assert value > 0, f"{model_name}.{field} must be positive, got {value}"


@pytest.mark.parametrize("model_name", list(BENCHMARK_METRICS.keys()))
def test_trainable_params_leq_total_params(model_name: str) -> None:
    """You can't train more parameters than the model contains."""
    row = BENCHMARK_METRICS[model_name]
    assert row["trainable_params"] <= row["total_params"]


@pytest.mark.parametrize("model_name", list(BENCHMARK_METRICS.keys()))
def test_normalization_is_cifar_or_imagenet(model_name: str) -> None:
    assert BENCHMARK_METRICS[model_name]["normalization"] in {"cifar", "imagenet"}


def test_best_model_key_is_in_registry() -> None:
    """``best_model_key()`` returns an actual entry in BENCHMARK_METRICS."""
    best = best_model_key()
    assert best in BENCHMARK_METRICS


def test_best_model_is_actually_best() -> None:
    """``best_model_key`` should win the max-accuracy contest."""
    best = best_model_key()
    best_acc = BENCHMARK_METRICS[best]["test_accuracy"]
    max_acc = max(m["test_accuracy"] for m in BENCHMARK_METRICS.values())
    assert best_acc == max_acc


def test_available_models_is_subset_of_benchmark() -> None:
    avail = available_models()
    assert set(avail) <= set(BENCHMARK_METRICS.keys())
    for name in avail:
        assert BENCHMARK_METRICS[name]["available"] is True


def test_accuracy_delta_matches_manual_subtraction() -> None:
    """Self-consistency: ``accuracy_delta(a, b)`` equals ``acc(a) - acc(b)``."""
    keys = list(BENCHMARK_METRICS.keys())
    if len(keys) < 2:
        pytest.skip("Need at least two models to test deltas")
    a, b = keys[0], keys[1]
    delta = accuracy_delta(a, b)
    expected = (
        BENCHMARK_METRICS[a]["test_accuracy"]
        - BENCHMARK_METRICS[b]["test_accuracy"]
    )
    assert abs(delta - expected) < 1e-9


# ----- Convergence history ------------------------------------------------


def test_convergence_history_only_references_known_models() -> None:
    """Every key in CONVERGENCE_HISTORY must exist in BENCHMARK_METRICS."""
    for name in CONVERGENCE_HISTORY:
        assert name in BENCHMARK_METRICS, f"Unknown model in convergence: {name}"


def test_convergence_histories_are_non_empty() -> None:
    for name, history in CONVERGENCE_HISTORY.items():
        assert len(history) > 0, f"{name}: empty convergence history"
        assert all(0.0 <= v <= 100.0 for v in history)


def test_convergence_histories_are_same_length() -> None:
    """The line chart relies on every series having the same epoch axis."""
    lengths = {len(h) for h in CONVERGENCE_HISTORY.values()}
    assert len(lengths) == 1, f"Mismatched lengths: {lengths}"


def test_convergence_final_matches_benchmark_accuracy() -> None:
    """The last convergence point should match the published test accuracy
    (within a small tolerance) for every model that has history."""
    for name, history in CONVERGENCE_HISTORY.items():
        final = history[-1]
        benchmark = BENCHMARK_METRICS[name]["test_accuracy"]
        assert abs(final - benchmark) < 0.5, (
            f"{name}: final epoch {final} vs published {benchmark}"
        )


# ----- Confusion pairs ----------------------------------------------------


def test_confusion_pairs_are_well_formed() -> None:
    assert len(CONFUSION_PAIRS) > 0
    for entry in CONFUSION_PAIRS:
        assert {"pair", "before", "after", "reduction_pct"} <= set(entry.keys())
        assert entry["before"] >= 0
        assert entry["after"] >= 0
        assert 0 <= entry["reduction_pct"] <= 100


def test_confusion_pair_reductions_are_consistent() -> None:
    """reduction_pct should broadly match (before - after) / before * 100.

    We allow a +/- 5-pt slack because the stored value is rounded to the
    nearest integer and the confusion matrices themselves were computed
    on different random seeds to the app's current checkpoint.
    """
    for entry in CONFUSION_PAIRS:
        if entry["before"] == 0:
            continue
        calculated = 100 * (entry["before"] - entry["after"]) / entry["before"]
        assert abs(calculated - entry["reduction_pct"]) <= 5, entry


# ----- Training config ----------------------------------------------------


def test_training_config_has_required_keys() -> None:
    required = {
        "seed", "epochs", "batch_size", "learning_rate", "weight_decay",
        "optimizer", "loss_function", "scheduler",
        "train_size", "test_size", "num_classes", "dataset",
    }
    assert required <= set(TRAINING_CONFIG.keys())


def test_training_config_cifar10_totals() -> None:
    """CIFAR-10 has 60 000 images split 50k / 10k — these totals are canonical."""
    assert TRAINING_CONFIG["num_classes"] == 10
    assert TRAINING_CONFIG["train_size"] == 50_000
    assert TRAINING_CONFIG["test_size"] == 10_000
    assert TRAINING_CONFIG["dataset"] == "CIFAR-10"
