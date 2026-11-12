"""
File: exceptions.py

Description:
Typed exception hierarchy for the application.

Responsibilities:
- Provide a single catchable base exception.
- Distinguish configuration, logging, startup, Ollama, and model failures.
"""


class BenchmarkError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(BenchmarkError):
    """Raised when configuration files are missing, malformed, or invalid."""


class LoggingError(BenchmarkError):
    """Raised when logging cannot be configured."""


class StartupError(BenchmarkError):
    """Raised when runtime directories or application initialization fail."""


class DatasetError(BenchmarkError):
    """Base exception for dataset loading and export failures."""


class DatasetValidationError(DatasetError):
    """Raised when a dataset, probe, or item fails validation."""


class OllamaError(BenchmarkError):
    """Base exception for Ollama client and model management failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""


class OllamaHTTPError(OllamaError):
    """Raised when Ollama returns an HTTP error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Ollama HTTP {status_code}: {detail}")


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns a malformed or incomplete response."""


class InvalidModelTagError(OllamaError):
    """Raised when a model tag is empty or syntactically invalid."""


class ModelNotFoundError(OllamaError):
    """Raised when a requested model is not installed."""


class ModelRegistrationError(OllamaError):
    """Raised when a model cannot be registered."""


class ModelNotRegisteredError(OllamaError):
    """Raised when a model is not part of the configured registry."""


class PullApprovalRequiredError(OllamaError):
    """Raised when a model pull is attempted without explicit approval."""


class ModelPullError(OllamaError):
    """Raised when Ollama reports a pull failure."""


class ModelLifecycleError(OllamaError):
    """Raised when a model load or unload operation fails."""


class ModelDigestMismatchError(OllamaError):
    """Raised when an installed model digest does not match configuration."""


class ModelQuantizationMismatchError(OllamaError):
    """Raised when an installed model quantization does not match configuration."""


class InferenceError(OllamaError):
    """Raised when a test inference request fails or returns an invalid result."""


class InferenceTimeoutError(InferenceError):
    """Raised when a streaming inference request exceeds its timeout."""


class StreamParseError(InferenceError):
    """Raised when a streaming response is malformed or incomplete."""


class EmptyResponseError(InferenceError):
    """Raised when a completed inference response contains no content."""


class InferenceCancelledError(InferenceError):
    """Raised when an inference request is cancelled by the caller."""


class EvaluationError(BenchmarkError):
    """Base exception for deterministic evaluation failures."""


class EvaluationConfigurationError(EvaluationError):
    """Raised when an item cannot be evaluated with its configured method."""


class SandboxError(EvaluationError):
    """Raised when the restricted code runner cannot execute a test."""


class JudgeError(EvaluationError):
    """Base exception for judge client, verification, and parsing failures."""


class JudgeVerificationError(JudgeError):
    """Raised when the configured judge artifact cannot be verified."""


class JudgeParseError(JudgeError):
    """Raised when judge output does not match the required JSON schema."""


class DatabaseError(BenchmarkError):
    """Base exception for database initialization and persistence failures."""


class DatabaseDuplicateError(DatabaseError):
    """Raised when an existing historical record would be overwritten."""


class DatabaseConflictError(DatabaseError):
    """Raised when a record conflicts with a stored dataset or experiment identity."""


class ExperimentStateError(DatabaseError):
    """Raised when an operation is not allowed in the current experiment state."""


class AnalyticsError(BenchmarkError):
    """Base exception for analytics, scoring, and recommendation failures."""


class InsufficientDataError(AnalyticsError):
    """Raised when stored experiment data cannot support the requested analysis."""


class ReportError(BenchmarkError):
    """Raised when a report or export cannot be generated safely."""
