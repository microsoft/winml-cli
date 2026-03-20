# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections.abc import Callable
from typing import TypeVar

from .base import NodeChecker


T = TypeVar("T")


class NodeCheckerRegistry:
    """Central registry for custom node checkers.

    Enables decorator-based registration of custom checker classes.
    """

    _checkers: dict[str, type[NodeChecker]] = {}

    @classmethod
    def register_checker(cls, name: str | None = None) -> Callable[[T], T]:
        """Decorator to register a custom checker class.

        Args:
            name: Optional name to register the checker under. If not provided,
                uses the class's __name__ attribute.

        Returns:
            Decorator function that registers the class and returns it unchanged

        Example:
            @NodeCheckerRegistry.register_checker()
            class BiasGeLUChecker(NodeChecker):
                pass

            @NodeCheckerRegistry.register_checker("custom_gelu")
            class MyGeLUChecker(NodeChecker):
                pass
        """

        def decorator(checker_class: T) -> T:
            checker_name = name or checker_class.__name__
            cls._checkers[checker_name] = checker_class
            return checker_class

        return decorator

    @classmethod
    def get_all_checkers(cls) -> list[type[NodeChecker]]:
        """Get all registered checker classes.

        Returns:
            List of all registered checker classes
        """
        return list(cls._checkers.values())
