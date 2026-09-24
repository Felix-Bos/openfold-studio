"""Business errors raised by the domain and service layers.

Every error carries a message that can be shown to the user as-is. Views
translate them into flash messages (HTML pages) or JSON error responses (API);
the layers below never know how an error will be displayed.
"""


class DomainError(Exception):
    """Base class for expected, user-facing errors."""

    @property
    def message(self) -> str:
        return str(self)


class InvalidSequenceError(DomainError):
    """The submitted protein sequence is empty or contains invalid letters."""


class ComputeBusyError(DomainError):
    """A prediction or attention extraction is already using the GPU.

    `blocking` is the PredictionJob or AttentionRun currently running, so the
    caller can send the user to it.
    """

    def __init__(self, message: str, blocking):
        super().__init__(message)
        self.blocking = blocking


class JobNotReadyError(DomainError):
    """The action requires a completed prediction job."""


class ResourceNotFoundError(DomainError):
    """A requested file, layer family, block or sample does not exist."""
