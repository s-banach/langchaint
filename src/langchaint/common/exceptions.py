"""Basic shared exceptions without langchaint imports."""


class TransientError(Exception):
    """One failed attempt that a retry may fix.

    `__cause__` holds the original provider exception when one exists.
    Retry loops raise `TransientError` inside `SharedBackoff.admitted()`.
    `SettledAttemptRecord.error` preserves normalized failure data.
    `SettledAttemptRecord.billing` preserves billing from the same request.
    """

    retry_after_seconds: float | None
    is_rate_limit: bool

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        is_rate_limit: bool = False,
    ) -> None:
        """Store the server-stated wait and rate-limit classification."""
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.is_rate_limit = is_rate_limit


class EmbeddingOutputError(RuntimeError):
    """A provider returned unusable embedding vectors."""


class StreamProtocolError(Exception):
    """A stream did not follow the event contract.

    A stream that ends without a terminal result raises this error.
    A missing Messages API stop reason or Responses API terminal response raises this error.
    A `StreamHandle` that ends without an adapter stream raises this error.
    `AdapterStream.final()` may raise this error before `AdapterStream.items()` is exhausted.
    """


class GaveUpWaiting(Exception):  # noqa: N818
    """A budget expired before `SharedBackoff.admitted()` admitted the request.

    The admission holds no permit or queue position and records no request.
    A new attempt joins the same queue behind the same pause.
    """


class ParserContractError(Exception):
    """A `SharedBackoff` parse function raised.

    This error identifies a defect in `parse` instead of a provider classification.
    `__cause__` holds the exception from `parse`.
    The provider failure passed to `parse` remains as exception context.
    `SharedBackoff` records no request outcome for this error.
    """
