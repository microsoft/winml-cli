# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Inject INSTRUMENTATION_KEY from env into constants.py before wheel build.

Called by the official build pipeline (`modelkit-official-build.yml`) just
before `python -m build`. Reads the key from env var `INSTRUMENTATION_KEY`
(mapped from the pipeline's secret variable) and replaces the empty
placeholder in `src/winml/modelkit/telemetry/constants.py`.

Fails loudly if the placeholder is missing or the key env var is empty —
better to break the build than ship a wheel with no telemetry.

The key value is NEVER printed to stdout/stderr to avoid leaking it
into pipeline logs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_TARGET = Path("src/winml/modelkit/telemetry/constants.py")
_PLACEHOLDER = 'INSTRUMENTATION_KEY = ""'


def main() -> int:
    """Replace the placeholder iKey in constants.py; return 0 on success."""
    key = os.environ.get("INSTRUMENTATION_KEY", "")
    if not key:
        print(
            "ERROR: INSTRUMENTATION_KEY env var is empty or missing. "
            "Map the pipeline secret variable explicitly via `env:`.",
            file=sys.stderr,
        )
        return 1

    if not _TARGET.exists():
        print(f"ERROR: target file {_TARGET} not found", file=sys.stderr)
        return 1

    src = _TARGET.read_text(encoding="utf-8")
    if _PLACEHOLDER not in src:
        print(
            f"ERROR: placeholder {_PLACEHOLDER!r} not found in {_TARGET}. "
            "The placeholder format may have drifted; update this script.",
            file=sys.stderr,
        )
        return 1

    new = src.replace(_PLACEHOLDER, f'INSTRUMENTATION_KEY = "{key}"')
    _TARGET.write_text(new, encoding="utf-8")

    # Re-read and confirm placeholder is gone — defense in depth against
    # filesystem oddities or partial writes.
    written = _TARGET.read_text(encoding="utf-8")
    if _PLACEHOLDER in written:
        print(
            f"ERROR: placeholder still present in {_TARGET} after write",
            file=sys.stderr,
        )
        return 1

    print(f"Injected iKey into {_TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
