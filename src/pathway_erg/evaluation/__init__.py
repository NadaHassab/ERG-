"""Evaluation utilities: fold-safe metrics and cluster bootstrap."""

from .metrics import BootstrapResult, binary_metrics, cluster_bootstrap_ci

__all__ = ["BootstrapResult", "binary_metrics", "cluster_bootstrap_ci"]
