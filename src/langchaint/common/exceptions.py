"""Basic shared exceptions without langchaint imports."""


class TransientError(Exception):
    """One failed request that a retry may fix.

    `__cause__` holds the original provider exception when one exists.
    Retry loops raise `TransientError` inside `SharedBackoff.admitted()`.
    `SettledRequestRecord.error` preserves normalized failure data.
    `SettledRequestRecord.billing` preserves billing from the same request.
    `pauses_quota` is true when the failure pauses every request on the rate-limit quota.
    """

    retry_after_seconds: float | None
    pauses_quota: bool

    def __init__(
        self,
        error_text: str,
        *,
        retry_after_seconds: float | None = None,
        pauses_quota: bool = False,
    ) -> None:
        """Store the server-stated wait and whether the failure pauses the rate-limit quota."""
        super().__init__(error_text)
        self.retry_after_seconds = retry_after_seconds
        self.pauses_quota = pauses_quota


class EmbeddingOutputError(RuntimeError):
    """A provider returned unusable embedding vectors."""


class StreamProtocolError(Exception):
    """A stream did not follow the event contract.

    A stream that ends without a terminal event raises this error.
    A missing Messages API stop reason or Responses API terminal response raises this error.
    `StreamHandle` reports this error from `AdapterStream.items()` as a `retry_unavailable_error` `GenerationError`.
    `generate_one` retries this error as a transient failure.
    `AdapterStream.final()` may raise this error before `AdapterStream.items()` is exhausted.
    """


class GaveUpWaitingError(Exception):
    """A budget expired before `SharedBackoff.admitted()` admitted the request.

    The admission holds no permit or queue position and records no request.
    A new request joins the same queue behind the same pause.
    """
