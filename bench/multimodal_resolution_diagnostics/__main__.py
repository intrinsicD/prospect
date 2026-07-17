"""Command-line entry point for the sealed MM-007 lifecycle."""

from .experiment import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
