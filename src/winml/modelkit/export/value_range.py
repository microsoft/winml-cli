"""Value Range Interceptor for Optimum's Dummy Input Generation.

Captures exact (min, max) value ranges per input by monkey-patching
DummyInputGenerator's static tensor generation methods during
onnx_config.generate_dummy_inputs(). This avoids copying Optimum's
internal range logic — ranges are captured from the actual calls.

Example:
    >>> from winml.modelkit.export.value_range import intercept_value_ranges
    >>> with intercept_value_ranges() as ranges:
    ...     onnx_config.generate_dummy_inputs(framework="pt")
    >>> ranges
    {'input_ids': {'min': 0, 'max': 30522, 'method': 'random_int_tensor'},
     'attention_mask': {'min': 0, 'max': 2, 'method': 'random_mask_tensor'}}
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import wraps

from optimum.utils.input_generators import DummyInputGenerator


# Thread-local to correlate static method calls with the current input_name.
# generate() wrapper sets .name before calling the original, static method
# wrappers read it to associate captured ranges with the correct input.
_current_input = threading.local()

# Static methods on DummyInputGenerator that produce tensors
_TENSOR_GEN_METHODS = (
    "random_int_tensor",
    "random_float_tensor",
    "random_mask_tensor",
    "constant_tensor",
)


def _make_static_wrapper(original, method_name, captured):
    """Wrap a DummyInputGenerator static method to capture value range args."""

    @wraps(original)
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        input_name = getattr(_current_input, "name", None)
        if input_name is None:
            return result

        # Extract (min, max) from the method's arguments
        if method_name == "random_int_tensor":
            # Signature: (shape, max_value, min_value=0, ...)
            lo = kwargs.get("min_value", args[2] if len(args) > 2 else 0)
            hi = kwargs.get("max_value", args[1] if len(args) > 1 else 0)
        elif method_name == "random_float_tensor":
            # Signature: (shape, min_value=0, max_value=1, ...)
            lo = kwargs.get("min_value", args[1] if len(args) > 1 else 0)
            hi = kwargs.get("max_value", args[2] if len(args) > 2 else 1)
        elif method_name == "random_mask_tensor":
            lo, hi = 0, 2  # binary {0, 1}, exclusive high
        elif method_name == "constant_tensor":
            val = kwargs.get("value", args[1] if len(args) > 1 else 1)
            lo, hi = val, val
        else:
            return result

        # Intersection: tighten range if multiple calls per input
        if input_name in captured:
            prev = captured[input_name]
            prev["min"] = max(prev["min"], lo)
            prev["max"] = min(prev["max"], hi)
        else:
            captured[input_name] = {"min": lo, "max": hi, "method": method_name}

        return result

    return wrapper


def _make_generate_wrapper(original):
    """Wrap a generator's generate() to track which input_name is active."""

    @wraps(original)
    def wrapper(self, input_name, *args, **kwargs):
        _current_input.name = input_name
        try:
            return original(self, input_name, *args, **kwargs)
        finally:
            _current_input.name = None

    return wrapper


@contextmanager
def intercept_value_ranges():
    """Context manager that captures value ranges from Optimum's dummy input generation.

    Monkey-patches DummyInputGenerator's static tensor methods and all
    subclass generate() methods to capture exact (min, max) per input name.

    For inputs where generate() calls multiple tensor methods (e.g.,
    past_key_values), ranges are intersected (tightest bounds).

    Yields:
        dict[str, dict] mapping input_name to {"min": lo, "max": hi, "method": name}.
        For integer inputs, max is exclusive (torch.randint semantics).

    Example:
        >>> with intercept_value_ranges() as ranges:
        ...     onnx_config.generate_dummy_inputs(framework="pt")
        >>> ranges
        {'input_ids': {'min': 0, 'max': 30522, 'method': 'random_int_tensor'},
         'attention_mask': {'min': 0, 'max': 2, 'method': 'random_mask_tensor'},
         'token_type_ids': {'min': 0, 'max': 2, 'method': 'random_int_tensor'}}
    """
    captured: dict[str, dict] = {}
    originals: dict = {}

    # Patch static tensor gen methods on the base class
    for method_name in _TENSOR_GEN_METHODS:
        original = getattr(DummyInputGenerator, method_name)
        originals[method_name] = original
        setattr(
            DummyInputGenerator,
            method_name,
            staticmethod(_make_static_wrapper(original, method_name, captured)),
        )

    # Patch generate() on all subclasses that override it
    patched_classes = []

    def _patch_subclasses(base):
        for cls in base.__subclasses__():
            if "generate" in cls.__dict__:
                originals[(cls, "generate")] = cls.__dict__["generate"]
                cls.generate = _make_generate_wrapper(cls.__dict__["generate"])
                patched_classes.append(cls)
            _patch_subclasses(cls)

    _patch_subclasses(DummyInputGenerator)

    try:
        yield captured
    finally:
        # Restore all originals — must re-wrap as staticmethod
        for method_name in _TENSOR_GEN_METHODS:
            setattr(
                DummyInputGenerator,
                method_name,
                staticmethod(originals[method_name]),
            )
        for cls in patched_classes:
            cls.generate = originals[(cls, "generate")]
