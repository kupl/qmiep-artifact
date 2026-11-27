"""Reconstruction-check policy scoped to a synthesis call and its recursion."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_VALIDATE_RECONSTRUCTION: ContextVar[bool] = ContextVar(
    "qmiep_validate_reconstruction", default=True
)


def _validation_enabled() -> bool:
    return _VALIDATE_RECONSTRUCTION.get()


@contextmanager
def _validation_scope(validate: bool | None) -> Iterator[None]:
    """Inherit the enclosing call's policy when ``validate`` is omitted."""
    if validate is not None and not isinstance(validate, bool):
        raise TypeError("validate must be a bool or None")
    token = _VALIDATE_RECONSTRUCTION.set(
        _validation_enabled() if validate is None else validate
    )
    try:
        yield
    finally:
        _VALIDATE_RECONSTRUCTION.reset(token)
