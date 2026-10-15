"""Dataset errors that should stop a run with a path and a reason."""


class DataError(ValueError):
    """Raised when a recording cannot be parsed into the common sample schema."""
