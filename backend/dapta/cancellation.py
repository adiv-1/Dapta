"""Cooperative cancellation scoped to one factory stage, never the whole pod."""
from contextlib import contextmanager
from contextvars import ContextVar


class BuildCancelled(RuntimeError):
    pass


class BuildCleanupError(RuntimeError):
    pass


_check = ContextVar("dapta_build_cancellation", default=None)


def cancellation_check():
    return _check.get()


def check_cancelled():
    check = cancellation_check()
    if check is not None:
        check()


@contextmanager
def cancellation_scope(check):
    token = _check.set(check)
    try:
        yield
    finally:
        _check.reset(token)
