from gateway.reliability import BreakerState, CircuitBreaker, backoff_delay


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_breaker_opens_after_threshold_and_recovers():
    clk = Clock()
    b = CircuitBreaker(3, 10, clk)
    for _ in range(2):
        b.record_failure()
    assert b.state is BreakerState.CLOSED and b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN and not b.allow()
    clk.t = 10
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow()  # single probe
    assert not b.allow()
    b.record_success()
    assert b.state is BreakerState.CLOSED


def test_half_open_failure_reopens():
    clk = Clock()
    b = CircuitBreaker(1, 5, clk)
    b.record_failure()
    clk.t = 5
    assert b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN
    clk.t = 9
    assert not b.allow()


def test_success_resets_failure_count():
    b = CircuitBreaker(2, 5, Clock())
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state is BreakerState.CLOSED


def test_backoff_is_jittered_and_capped():
    assert backoff_delay(0, 0.2, 2, lambda: 1.0) == 0.2
    assert backoff_delay(10, 0.2, 2, lambda: 1.0) == 2
    assert backoff_delay(3, 0.2, 2, lambda: 0.0) == 0
