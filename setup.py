# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Select the native ARM64 runtime profile by wheel target, not host CPU."""

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class WindowsWheel(bdist_wheel):
    """Build the preview wheel without the x64 export and evaluation stack."""

    def run(self) -> None:
        """Use the reduced dependency set only for an explicitly tagged ARM64 wheel."""
        if self.plat_name == "win_arm64" and self.plat_name_supplied:
            runtime = {
                "click",
                "colorama",
                "filelock",
                "httpx",
                "jsonschema",
                "ml-dtypes",
                "numpy",
                "onnx",
                "onnxruntime-genai-winml",
                "onnxruntime-windowsml",
                "opentelemetry-sdk",
                "plotext",
                "psutil",
                "pydantic",
                "requests",
                "rich",
                "snakemd",
                "tqdm",
                "windowsml",
            }
            dependencies = []
            for req in map(Requirement, self.distribution.install_requires):
                if req.name not in runtime:
                    continue
                if req.name == "windowsml":
                    req.extras.add("llama")
                elif req.name == "ml-dtypes":
                    req.specifier = SpecifierSet("==0.5.4")
                dependencies.append(str(req))
            self.distribution.install_requires = dependencies
            self.distribution.metadata.install_requires = dependencies
        super().run()


setup(cmdclass={"bdist_wheel": WindowsWheel})
