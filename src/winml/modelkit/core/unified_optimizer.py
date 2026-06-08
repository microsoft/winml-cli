# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unified Optimization Framework for ModelExport.

This module provides a unified optimization framework that applies optimizations
across all export strategies based on learnings from iterations 17-18.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar

import torch


logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Profile of optimizations applied to an exporter."""

    strategy_name: str
    optimizations_applied: list[str]
    performance_metrics: dict[str, float]
    warnings: list[str]

    def get_summary(self) -> str:
        """Get a summary of applied optimizations."""
        return f"{self.strategy_name}: {len(self.optimizations_applied)} optimizations applied"


class PerformanceMonitor:
    """Monitor and track performance metrics across operations."""

    def __init__(self) -> None:
        self.timings: defaultdict[str, list[float]] = defaultdict(list)
        self.counters: defaultdict[str, int] = defaultdict(int)

    def time_operation(self, operation_name: str) -> Callable[..., Any]:
        """Decorator to time an operation."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    self.timings[operation_name].append(elapsed)
                    return result
                except Exception:
                    elapsed = time.time() - start_time
                    self.timings[f"{operation_name}_error"].append(elapsed)
                    raise

            return wrapper

        return decorator

    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """Increment a performance counter."""
        self.counters[counter_name] += value

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        metrics: dict[str, Any] = {}

        # Timing metrics
        for operation, times in self.timings.items():
            if times:
                metrics[f"{operation}_count"] = len(times)
                metrics[f"{operation}_total"] = sum(times)
                metrics[f"{operation}_avg"] = sum(times) / len(times)
                metrics[f"{operation}_min"] = min(times)
                metrics[f"{operation}_max"] = max(times)

        # Counter metrics
        metrics.update(self.counters)

        return metrics


class UnifiedOptimizer:
    """Unified optimization framework for all export strategies.

    Applies common optimizations learned from iterations 17-18.
    """

    # Common optimizations that apply to all strategies
    COMMON_OPTIMIZATIONS: ClassVar[dict[str, dict[str, Any]]] = {
        "single_pass_algorithms": {
            "description": "Use single-pass algorithms to reduce redundant computation",
            "applicable_to": ["all"],
            "impact": "medium",
        },
        "batch_processing": {
            "description": "Batch similar operations together",
            "applicable_to": ["all"],
            "impact": "medium",
        },
        "caching": {
            "description": "Cache computed values to avoid recomputation",
            "applicable_to": ["all"],
            "impact": "high",
        },
        "lightweight_operations": {
            "description": "Use lightweight data structures and operations",
            "applicable_to": ["all"],
            "impact": "medium",
        },
        "optimized_onnx_params": {
            "description": "Optimize ONNX export parameters",
            "applicable_to": ["all"],
            "impact": "medium",
        },
    }

    # Strategy-specific optimizations
    STRATEGY_OPTIMIZATIONS: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "htp": {
            "tag_injection_optimization": {
                "description": "Optimize tag injection using single-pass and Counter",
                "impact": "high",
            },
            "builtin_tracking": {
                "description": "Use PyTorch's built-in module tracking",
                "impact": "medium",
            },
        },
        "usage_based": {
            "lightweight_hooks": {
                "description": "Use minimal overhead forward hooks",
                "impact": "low",
            },
            "pre_allocated_structures": {
                "description": "Pre-allocate data structures",
                "impact": "low",
            },
        },
        "fx": {
            "graph_caching": {
                "description": "Cache FX graph transformations",
                "impact": "medium",
            },
            "node_batching": {
                "description": "Batch node operations in FX graph",
                "impact": "medium",
            },
        },
    }

    def __init__(self) -> None:
        self.monitor = PerformanceMonitor()
        self.applied_optimizations: list[str] = []

    def optimize_exporter(self, exporter: Any, strategy_name: str) -> OptimizationProfile:
        """Apply optimizations to an exporter based on its strategy.

        Args:
            exporter: The exporter instance to optimize
            strategy_name: Name of the export strategy

        Returns:
            OptimizationProfile with details of applied optimizations
        """
        profile = OptimizationProfile(
            strategy_name=strategy_name,
            optimizations_applied=[],
            performance_metrics={},
            warnings=[],
        )

        # Apply common optimizations
        self._apply_common_optimizations(exporter, profile)

        # Apply strategy-specific optimizations
        if strategy_name in self.STRATEGY_OPTIMIZATIONS:
            self._apply_strategy_optimizations(exporter, strategy_name, profile)

        # Add performance monitor
        if not hasattr(exporter, "_performance_monitor"):
            exporter._performance_monitor = self.monitor
            profile.optimizations_applied.append("performance_monitoring")

        # Log optimization summary
        opt_count = len(profile.optimizations_applied)
        logger.info(f"Applied {opt_count} optimizations to {strategy_name} exporter")

        return profile

    def _apply_common_optimizations(
        self, exporter: Any, profile: OptimizationProfile
    ) -> None:
        """Apply optimizations common to all strategies."""
        # 1. Optimize ONNX export parameters
        if hasattr(exporter, "export"):
            original_export = exporter.export

            @wraps(original_export)
            def optimized_export(
                model: Any, example_inputs: Any, output_path: str, **kwargs: Any
            ) -> Any:
                # Apply ONNX parameter optimizations
                import torch

                kwargs.setdefault("training", torch.onnx.TrainingMode.EVAL)
                kwargs.setdefault("opset_version", 14)
                kwargs.setdefault("verbose", False)
                kwargs.setdefault("operator_export_type", torch.onnx.OperatorExportTypes.ONNX)
                kwargs.setdefault("keep_initializers_as_inputs", True)

                return original_export(model, example_inputs, output_path, **kwargs)

            exporter.export = optimized_export
            profile.optimizations_applied.append("optimized_onnx_params")

        # 2. Add caching capability
        if not hasattr(exporter, "_cache"):
            exporter._cache = {}
            exporter._cache_hits = 0
            exporter._cache_misses = 0

            def get_cached(key: str, compute_func: Callable[[], Any]) -> Any:
                """Get value from cache or compute it."""
                if key in exporter._cache:
                    exporter._cache_hits += 1
                    return exporter._cache[key]
                exporter._cache_misses += 1
                value = compute_func()
                exporter._cache[key] = value
                return value

            exporter.get_cached = get_cached
            profile.optimizations_applied.append("caching")

        # 3. Add batch processing utilities
        if not hasattr(exporter, "batch_process"):

            def batch_process(
                items: list[Any], process_func: Callable[[Any], Any], batch_size: int = 100
            ) -> list[Any]:
                """Process items in batches for efficiency."""
                results = []
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    batch_results = [process_func(item) for item in batch]
                    results.extend(batch_results)
                return results

            exporter.batch_process = batch_process
            profile.optimizations_applied.append("batch_processing")

    def _apply_strategy_optimizations(
        self, exporter: Any, strategy_name: str, profile: OptimizationProfile
    ) -> None:
        """Apply strategy-specific optimizations."""
        if strategy_name == "htp":
            # Apply HTP-specific optimizations from iteration 17
            try:
                from ..strategies.htp.optimizations import (  # type: ignore[import-not-found]
                    apply_htp_optimizations,
                )

                apply_htp_optimizations(exporter)
                profile.optimizations_applied.extend(
                    ["tag_injection_optimization", "builtin_tracking"]
                )
            except ImportError:
                profile.warnings.append("HTP optimizations module not found")

        elif strategy_name == "usage_based":
            # Apply Usage-Based optimizations from iteration 18
            try:
                from ..strategies.usage_based.optimizations import (  # type: ignore[import-not-found]
                    apply_usage_based_optimizations,
                )

                apply_usage_based_optimizations(exporter)
                profile.optimizations_applied.extend(
                    ["lightweight_hooks", "pre_allocated_structures"]
                )
            except ImportError:
                profile.warnings.append("Usage-Based optimizations module not found")

        elif strategy_name == "fx":
            # Apply FX-specific optimizations
            self._apply_fx_optimizations(exporter, profile)

    def _apply_fx_optimizations(
        self, exporter: Any, profile: OptimizationProfile
    ) -> None:
        """Apply FX-specific optimizations."""
        # Add graph caching for FX
        if hasattr(exporter, "_trace_transformers_model"):
            original_trace = exporter._trace_transformers_model
            graph_cache: dict[int, Any] = {}

            @wraps(original_trace)
            def cached_trace(
                model: Any, example_inputs: Any = None
            ) -> Any:
                model_id = id(model)
                if model_id in graph_cache:
                    logger.debug("Using cached FX graph")
                    return graph_cache[model_id]

                result = original_trace(model, example_inputs)
                graph_cache[model_id] = result
                return result

            exporter._trace_transformers_model = cached_trace
            profile.optimizations_applied.append("graph_caching")


def create_optimized_exporter(strategy: str, **kwargs: Any) -> Any:
    """Create an optimized exporter for the given strategy.

    Args:
        strategy: Export strategy name ("usage_based", "htp", "fx_graph")
        **kwargs: Additional arguments for the exporter

    Returns:
        Optimized exporter instance
    """
    # Import strategy modules
    if strategy == "usage_based":
        from ..strategies.usage_based import UsageBasedExporter  # type: ignore[import-not-found]

        exporter = UsageBasedExporter(**kwargs)
    elif strategy == "htp":
        from ..strategies.htp import HTPExporter  # type: ignore[import-not-found]

        exporter = HTPExporter(**kwargs)
    elif strategy == "fx_graph" or strategy == "fx":
        from ..strategies.fx import FXHierarchyExporter  # type: ignore[import-not-found]

        exporter = FXHierarchyExporter(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply unified optimizations
    optimizer = UnifiedOptimizer()
    optimization_profile = optimizer.optimize_exporter(exporter, strategy)

    # Store optimization profile
    exporter._optimization_profile = optimization_profile

    opt_count = len(optimization_profile.optimizations_applied)
    logger.info(f"Created optimized {strategy} exporter with {opt_count} optimizations")

    return exporter


class OptimizationBenchmark:
    """Benchmark the impact of optimizations."""

    @staticmethod
    def compare_optimized_vs_original(
        model: torch.nn.Module, example_inputs: Any, strategy: str, num_runs: int = 3
    ) -> dict[str, Any]:
        """Compare optimized vs original exporter performance.

        Returns:
            Dictionary with comparison metrics
        """
        import tempfile
        from pathlib import Path

        results: dict[str, Any] = {
            "strategy": strategy,
            "original_times": [],
            "optimized_times": [],
            "optimization_impact": {},
        }

        # Test original
        for _i in range(num_runs):
            # Create unoptimized exporter
            if strategy == "usage_based":
                from ..strategies.usage_based import UsageBasedExporter

                exporter = UsageBasedExporter()
            elif strategy == "htp":
                from ..strategies.htp import HTPExporter

                exporter = HTPExporter()
            else:
                continue

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                start_time = time.time()
                exporter.export(model, example_inputs, tmp.name)
                elapsed = time.time() - start_time
                results["original_times"].append(elapsed)
                Path(tmp.name).unlink()

        # Test optimized
        for _i in range(num_runs):
            exporter = create_optimized_exporter(strategy)

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                start_time = time.time()
                exporter.export(model, example_inputs, tmp.name)
                elapsed = time.time() - start_time
                results["optimized_times"].append(elapsed)
                Path(tmp.name).unlink()

        # Calculate impact
        avg_original = sum(results["original_times"]) / len(results["original_times"])
        avg_optimized = sum(results["optimized_times"]) / len(results["optimized_times"])

        results["optimization_impact"] = {
            "avg_original": avg_original,
            "avg_optimized": avg_optimized,
            "improvement": (avg_original - avg_optimized) / avg_original * 100,
            "speedup": avg_original / avg_optimized,
        }

        return results
