"""Cellpose reference workflow for Odon."""

from .extension import CellposeExtension


def main() -> None:
    from odon.extensions import run

    run(CellposeExtension, reconnect=True)

__all__ = ["CellposeExtension", "main"]
__version__ = "0.1.0"
