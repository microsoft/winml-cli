# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Table utilities for runtime rule tables."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_table_df(raw_table: Any) -> pd.DataFrame:
    """Build DataFrame from runtime table JSON while preserving int/None types.

    Runtime table JSON is stored as columnar dicts. Building with ``from_dict`` can
    upcast int+None columns to float (for example, ``0/1/None`` to
    ``0.0/1.0/nan``). Convert to row records first and force object dtype to keep
    exact Python values.
    """
    if not isinstance(raw_table, dict) or not raw_table:
        return pd.DataFrame()

    columns = list(raw_table.keys())
    first_col = raw_table[columns[0]]

    if isinstance(first_col, dict):
        row_keys = sorted(first_col.keys(), key=int)
        rows = [{c: raw_table[c].get(k) for c in columns} for k in row_keys]
    else:
        row_count = len(first_col)
        rows = [{c: raw_table[c][i] for c in columns} for i in range(row_count)]

    return pd.DataFrame(rows, dtype=object)
