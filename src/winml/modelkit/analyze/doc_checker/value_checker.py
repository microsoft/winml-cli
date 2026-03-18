"""Value Constraint Checker (ADR-001)

Validates parameter value constraints for QNN operators.
Used by both constraint validation and operator mapping systems.
"""

from typing import Any


class ValueConstraintChecker:
    """Parameter value constraint checker for QNN operators."""

    @staticmethod
    def check_allowed_values(value: Any, allowed: Any | list[Any]) -> tuple[bool, str]:
        """Check if value is in allowed set.
        
        Used extensively in operator mapping (ADR-002) to select QNN operators
        based on attribute values (e.g., Resize mode, Mod fmod).
        
        Supports both single value and list of allowed values for flexible matching.
        
        Args:
            value: Value to check
            allowed: Single allowed value or list of allowed values
            
        Returns:
            Tuple of (success, message)
            
        Examples:
            >>> # Single value check
            >>> ValueConstraintChecker.check_allowed_values("nearest", "nearest")
            (True, "OK")
            >>> ValueConstraintChecker.check_allowed_values("linear", "nearest")
            (False, "Value linear not in allowed set ['nearest']")
            
            >>> # Multi-value check
            >>> ValueConstraintChecker.check_allowed_values("cubic", ["cubic", ""])
            (True, "OK")
            >>> ValueConstraintChecker.check_allowed_values("nearest", ["cubic", ""])
            (False, "Value nearest not in allowed set ['cubic', '']")
        """
        # Normalize to list for unified processing
        allowed_list = allowed if isinstance(allowed, list) else [allowed]

        if value not in allowed_list:
            return False, f"Value {value} not in allowed set {allowed_list}"
        return True, "OK"

    @staticmethod
    def check_range(value: int | float, min_val: int | float,
                    max_val: int | float) -> tuple[bool, str]:
        """Check if value is within range.
        
        Args:
            value: Value to check
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            
        Returns:
            Tuple of (success, message)
            
        Examples:
            >>> ValueConstraintChecker.check_range(5, 0, 10)
            (True, "OK")
            >>> ValueConstraintChecker.check_range(15, 0, 10)
            (False, "Value 15 not in range [0, 10]")
        """
        if not (min_val <= value <= max_val):
            return False, f"Value {value} not in range [{min_val}, {max_val}]"
        return True, "OK"
