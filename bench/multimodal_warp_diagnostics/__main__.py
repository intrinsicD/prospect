"""Command-line entry point for the sealed MM-006 lifecycle."""

from .experiment import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
