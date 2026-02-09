"""Executable entrypoint for `python -m dlgforge`.

Delegates directly to :func:`dlgforge.cli.main`.
"""

from dlgforge.cli import main

if __name__ == "__main__":
    main()
