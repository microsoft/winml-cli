# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shape constraint checker for QNN operators.

This module provides static methods for validating tensor shape and rank
constraints as defined in QNN HTP Backend documentation. Handles the most
common constraint type (41.9% of all QNN constraints).
"""


class ShapeConstraintChecker:
    """Shape constraint validation for QNN operators.

    Provides static methods for validating tensor shapes against QNN
    constraints including rank limits, exact ranks, dimension ranges,
    and dimension-specific validations.

    All methods return (success, message) tuples:
    - success (bool): True if constraint satisfied, False otherwise
    - message (str): "OK" on success, descriptive error message on failure

    Example:
        >>> from shape_checker import ShapeConstraintChecker
        >>> shape = [1, 3, 224, 224]
        >>> success, msg = ShapeConstraintChecker.check_max_rank(shape, 4)
        >>> assert success, msg
    """

    @staticmethod
    def check_max_rank(shape: list[int], max_rank: int) -> tuple[bool, str]:
        """Check if tensor rank is within maximum limit.

        Validates the most common QNN shape constraint: maximum supported
        rank. Used by 596+ operators in QNN HTP Backend.

        Args:
            shape: Tensor shape as list of dimensions (e.g., [1, 3, 224, 224])
            max_rank: Maximum allowed rank for this operator/configuration

        Returns:
            Tuple of (success, message):
            - (True, "OK") if rank <= max_rank
            - (False, "Rank X exceeds max Y") otherwise

        Example:
            >>> ShapeConstraintChecker.check_max_rank([1, 3, 224, 224], 4)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_max_rank([1, 3, 16, 112, 112], 4)
            (False, 'Rank 5 exceeds max 4')

        Note:
            Common QNN max_rank values: 4 (Conv2d, MaxPool), 5 (Conv3d, Cast)
        """
        rank = len(shape)
        if rank > max_rank:
            return False, f"Rank {rank} exceeds max {max_rank}"
        return True, "OK"

    @staticmethod
    def check_min_rank(shape: list[int], min_rank: int) -> tuple[bool, str]:
        """Check if tensor rank meets minimum requirement.

        Validates minimum rank constraints for operators that require
        at least N dimensions.

        Args:
            shape: Tensor shape as list of dimensions
            min_rank: Minimum required rank

        Returns:
            Tuple of (success, message):
            - (True, "OK") if rank >= min_rank
            - (False, "Rank X below minimum Y") otherwise

        Example:
            >>> ShapeConstraintChecker.check_min_rank([224, 224], 2)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_min_rank([224], 2)
            (False, 'Rank 1 below minimum 2')
        """
        rank = len(shape)
        if rank < min_rank:
            return False, f"Rank {rank} below minimum {min_rank}"
        return True, "OK"

    @staticmethod
    def check_exact_rank(shape: list[int], exact_rank: int) -> tuple[bool, str]:
        """Check if tensor has exact required rank.

        Validates strict rank requirements for operators that only support
        specific dimensionalities. Common for 3D convolutions (rank=5) and
        format-specific operators (NCHW rank=4).

        Args:
            shape: Tensor shape as list of dimensions
            exact_rank: Required rank value

        Returns:
            Tuple of (success, message):
            - (True, "OK") if rank == exact_rank
            - (False, "Rank X must be exactly Y") otherwise

        Example:
            >>> ShapeConstraintChecker.check_exact_rank([1, 3, 16, 112, 112], 5)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_exact_rank([1, 3, 224, 224], 5)
            (False, 'Rank 4 must be exactly 5')

        Note:
            Used by Conv3d (rank=5), weights (specific ranks per operator)
        """
        rank = len(shape)
        if rank != exact_rank:
            return False, f"Rank {rank} must be exactly {exact_rank}"
        return True, "OK"

    @staticmethod
    def check_rank_range(shape: list[int], min_rank: int, max_rank: int) -> tuple[bool, str]:
        """Check if tensor rank is within allowed range.

        Validates range constraints for operators supporting multiple
        dimensionalities. Common for flexible operators like BatchNorm
        (rank 1-4), ElementWise operations (rank 1-5).

        Args:
            shape: Tensor shape as list of dimensions
            min_rank: Minimum allowed rank (inclusive)
            max_rank: Maximum allowed rank (inclusive)

        Returns:
            Tuple of (success, message):
            - (True, "OK") if min_rank <= rank <= max_rank
            - (False, "Rank X must be between Y and Z") otherwise

        Raises:
            ValueError: If min_rank > max_rank

        Example:
            >>> ShapeConstraintChecker.check_rank_range([1, 3, 224, 224], 1, 4)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_rank_range([224], 2, 5)
            (False, 'Rank 1 must be between 2 and 5')

        Note:
            Combines min and max checks for efficiency
        """
        if min_rank > max_rank:
            raise ValueError(f"min_rank ({min_rank}) cannot be greater than max_rank ({max_rank})")

        rank = len(shape)
        if rank < min_rank or rank > max_rank:
            return False, f"Rank {rank} must be between {min_rank} and {max_rank}"
        return True, "OK"

    @staticmethod
    def check_dimension_value(
        shape: list[int], dim_index: int, expected_value: int
    ) -> tuple[bool, str]:
        """Check if specific dimension has expected value.

        Validates dimension-specific constraints such as channel count,
        spatial dimensions, or batch size requirements.

        Args:
            shape: Tensor shape as list of dimensions
            dim_index: Index of dimension to check (0-based, negative indexing supported)
            expected_value: Required value for this dimension

        Returns:
            Tuple of (success, message):
            - (True, "OK") if shape[dim_index] == expected_value
            - (False, "Dimension X is Y, expected Z") otherwise

        Raises:
            IndexError: If dim_index is out of bounds

        Example:
            >>> # Check NCHW format has 3 channels
            >>> ShapeConstraintChecker.check_dimension_value([1, 3, 224, 224], 1, 3)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_dimension_value([1, 64, 224, 224], 1, 3)
            (False, 'Dimension 1 is 64, expected 3')
        """
        try:
            actual_value = shape[dim_index]
        except IndexError:
            return False, f"Dimension index {dim_index} out of bounds for shape {shape}"

        if actual_value != expected_value:
            return False, f"Dimension {dim_index} is {actual_value}, expected {expected_value}"
        return True, "OK"

    @staticmethod
    def check_dimension_range(
        shape: list[int], dim_index: int, min_value: int, max_value: int
    ) -> tuple[bool, str]:
        """Check if specific dimension is within allowed range.

        Validates dimension value constraints such as "width between 16-512"
        or "channels 1-256". Useful for hardware-specific limitations.

        Args:
            shape: Tensor shape as list of dimensions
            dim_index: Index of dimension to check
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)

        Returns:
            Tuple of (success, message):
            - (True, "OK") if min_value <= shape[dim_index] <= max_value
            - (False, "Dimension X value Y must be between A and B") otherwise

        Raises:
            IndexError: If dim_index is out of bounds
            ValueError: If min_value > max_value

        Example:
            >>> # Check width (dim 3) is between 16 and 512
            >>> ShapeConstraintChecker.check_dimension_range([1, 3, 224, 256], 3, 16, 512)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_dimension_range([1, 3, 224, 8], 3, 16, 512)
            (False, 'Dimension 3 value 8 must be between 16 and 512')
        """
        if min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value ({max_value})"
            )

        try:
            actual_value = shape[dim_index]
        except IndexError:
            return False, f"Dimension index {dim_index} out of bounds for shape {shape}"

        if actual_value < min_value or actual_value > max_value:
            return False, (
                f"Dimension {dim_index} value {actual_value} "
                f"must be between {min_value} and {max_value}"
            )
        return True, "OK"

    @staticmethod
    def check_dimension_divisible(
        shape: list[int], dim_index: int, divisor: int
    ) -> tuple[bool, str]:
        """Check if dimension value is divisible by factor.

        Validates divisibility constraints common in convolution operators
        (e.g., "width must be divisible by 2") and channel requirements
        (e.g., "channels divisible by group size").

        Args:
            shape: Tensor shape as list of dimensions
            dim_index: Index of dimension to check
            divisor: Required divisor (must be > 0)

        Returns:
            Tuple of (success, message):
            - (True, "OK") if shape[dim_index] % divisor == 0
            - (False, "Dimension X value Y not divisible by Z") otherwise

        Raises:
            IndexError: If dim_index is out of bounds
            ValueError: If divisor <= 0

        Example:
            >>> # Check channels (dim 1) divisible by 8
            >>> ShapeConstraintChecker.check_dimension_divisible([1, 64, 224, 224], 1, 8)
            (True, 'OK')
            >>> ShapeConstraintChecker.check_dimension_divisible([1, 65, 224, 224], 1, 8)
            (False, 'Dimension 1 value 65 not divisible by 8')
        """
        if divisor <= 0:
            raise ValueError(f"divisor must be positive, got {divisor}")

        try:
            actual_value = shape[dim_index]
        except IndexError:
            return False, f"Dimension index {dim_index} out of bounds for shape {shape}"

        if actual_value % divisor != 0:
            return False, f"Dimension {dim_index} value {actual_value} not divisible by {divisor}"
        return True, "OK"

    @staticmethod
    def check_dimension_multiple(
        shape: list[int], dim_index: int, multiple_of: int
    ) -> tuple[bool, str]:
        """Check if dimension value is a multiple of a given factor.

        Alias for check_dimension_divisible for clearer semantics.

        Args:
            shape: Tensor shape as list of dimensions
            dim_index: Index of dimension to check
            multiple_of: Required factor

        Returns:
            Tuple of (success, message)
        """
        return ShapeConstraintChecker.check_dimension_divisible(shape, dim_index, multiple_of)
