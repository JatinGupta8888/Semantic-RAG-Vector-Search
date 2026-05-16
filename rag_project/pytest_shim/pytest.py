"""
pytest.py  — minimal shim so test files can `import pytest` without the real package.
Provides: fixture decorator (no-op), raises context manager, mark namespace.
"""
from __future__ import annotations
import contextlib
from typing import Any, Callable, Type


class _RaisesCtx:
    def __init__(self, exc_type, match=None):
        self._exc_type = exc_type
        self._match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self._exc_type.__name__} but no exception was raised.")
        if not issubclass(exc_type, self._exc_type):
            return False  # let it propagate
        if self._match:
            import re
            if not re.search(self._match, str(exc_val)):
                raise AssertionError(
                    f"Expected {self._exc_type.__name__} matching {self._match!r}, "
                    f"got: {exc_val!r}"
                )
        self.value = exc_val
        return True  # suppress


def raises(exc_type: Type[Exception], match: str | None = None) -> _RaisesCtx:
    return _RaisesCtx(exc_type, match)


def fixture(fn=None, *, scope="function", **kwargs):
    """No-op decorator — fixtures are resolved by run_tests.py directly."""
    def decorator(f):
        return f
    return decorator(fn) if fn is not None else decorator


class _Mark:
    def __getattr__(self, name):
        def decorator(fn=None, **kw):
            if fn is not None:
                return fn
            return lambda f: f
        return decorator


mark = _Mark()


def skip(reason=""):
    raise _SkipException(reason)


class _SkipException(Exception):
    pass


def approx(expected, rel=None, abs=None, nan_ok=False):
    return expected  # close enough for our tests
