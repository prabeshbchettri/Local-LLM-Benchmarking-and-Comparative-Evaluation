"""
File: __init__.py

Description:
Core application services for configuration, logging, startup, and health checks.

Responsibilities:
- Export the public core API.
"""

from benchmark.core.config import AppSettings, compute_config_hash, load_settings
from benchmark.core.exceptions import (
    BenchmarkError,
    ConfigurationError,
    InferenceError,
    InvalidModelTagError,
    LoggingError,
    ModelDigestMismatchError,
    ModelLifecycleError,
    ModelNotFoundError,
    ModelNotRegisteredError,
    ModelPullError,
    ModelQuantizationMismatchError,
    ModelRegistrationError,
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
    PullApprovalRequiredError,
    StartupError,
)
from benchmark.core.health import CheckResult, HealthReport, render_health_report, run_health_check
from benchmark.core.logging import configure_logging, get_logger, reset_logging
from benchmark.core.startup import ensure_runtime_directories, initialize_application

__all__ = [
    "AppSettings",
    "BenchmarkError",
    "CheckResult",
    "ConfigurationError",
    "HealthReport",
    "InferenceError",
    "InvalidModelTagError",
    "LoggingError",
    "ModelDigestMismatchError",
    "ModelLifecycleError",
    "ModelNotFoundError",
    "ModelNotRegisteredError",
    "ModelPullError",
    "ModelQuantizationMismatchError",
    "ModelRegistrationError",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaHTTPError",
    "OllamaResponseError",
    "OllamaTimeoutError",
    "PullApprovalRequiredError",
    "StartupError",
    "compute_config_hash",
    "configure_logging",
    "ensure_runtime_directories",
    "get_logger",
    "initialize_application",
    "load_settings",
    "render_health_report",
    "reset_logging",
    "run_health_check",
]
