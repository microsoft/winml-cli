# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Back-compat shim: run the SAM 3 preset of the generic mask-gen harness.

The original SAM 3 logic has been generalized into
:mod:`scripts.mask_generation_eval` so other promptable-segmentation
models (SAM 2 / SAM 2.1 / future ``onnx-community`` exports) can reuse
the same harness via ``--preset``.  This wrapper preserves the original
``python scripts/sam3_smoke_eval.py`` entrypoint for anyone with
bookmarks or CI invocations.

The old standalone script had a bug -- it used ImageNet mean/std for
SAM 3, but the SAM 3 Tracker image processor uses ``[0.5, 0.5, 0.5]``
for both mean and std (matches ``Sam3TrackerImageProcessor`` defaults).
The generic harness encodes the correct values in the ``sam3`` preset.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    # Delegate to the generic harness with the SAM 3 preset baked in.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mask_generation_eval import main  # type: ignore[import-not-found]

    # Inject ``--preset sam3`` if the user didn't override it.
    if "--preset" not in sys.argv and "--encoder" not in sys.argv:
        sys.argv.insert(1, "--preset")
        sys.argv.insert(2, "sam3")
    sys.exit(main())
