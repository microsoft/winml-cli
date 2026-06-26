# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Package data for WinML CLI.

This package ships the built-in model catalog (``hub_models.json``) consumed by
the ``catalog`` command and the ``serve`` HTTP API. It intentionally contains no
importable code so that resolving the package (e.g. via
``importlib.resources.files``) stays lightweight and free of heavy optional
dependencies.
"""
