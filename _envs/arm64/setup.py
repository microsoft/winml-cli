# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build the ARM distribution from the checkout's shared Python and package data."""

from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.sdist import sdist


try:
    import tomllib
except ModuleNotFoundError:  # The release build host can be older than the target interpreter.
    import tomli as tomllib


class WheelOnly(sdist):
    """The ARM project references sources outside its own directory."""

    def run(self) -> None:
        """Do not produce a partial source archive from this subproject."""
        raise ValueError("Use 'uv build --wheel' here; build the sdist from the repository root.")


root = Path(__file__).resolve().parents[2]
with (root / "pyproject.toml").open("rb") as stream:
    shared = tomllib.load(stream)

source = "../../src"
setup(
    version=shared["project"]["version"],
    long_description=(root / shared["project"]["readme"]).read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(where=source, include=["winml", "winml.*"]),
    package_dir={"": source},
    package_data=shared["tool"]["setuptools"]["package-data"],
    include_package_data=False,
    data_files=[("share/licenses/winml-cli", ["../../LICENSE.txt"])],
    cmdclass={"sdist": WheelOnly},
    options={"bdist_wheel": {"plat_name": "win_arm64"}, "egg_info": {"egg_base": "."}},
)
