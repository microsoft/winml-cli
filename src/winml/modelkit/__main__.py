"""Module execution entry point: python -m winml.modelkit.

This module enables running ModelKit CLI via Python module execution:
    python -m winml.modelkit --version
    python -m winml.modelkit export --model MODEL --output PATH
"""

from .cli import main


if __name__ == "__main__":
    main()
