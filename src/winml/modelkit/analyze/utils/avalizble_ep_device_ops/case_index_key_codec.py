# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any


ALPHABET = set("123456789abcdefghijklmnopqrstuvwxyz")
DEVICE_SET = {"CPU", "GPU", "NPU"}


@dataclass(frozen=True)
class DecodedLocation:
    """Decoded folder/file location for a case_index key prefix."""

    folder_name: str
    file_name: str


IndexBundle = tuple[
    dict[tuple[str, str], str],
    dict[str, tuple[str, str]],
    dict[tuple[str, int, str], str],
    dict[tuple[str, int], list[tuple[str, str]]],
    dict[str, tuple[str, int, str]],
]


def _default_mapping_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    return base_dir / "avaliable_providers.json", base_dir / "available_ops_all.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid JSON root object: {path}")
    return payload


def _normalize_stem(file_name: str) -> tuple[str, bool]:
    stem = Path(file_name).name
    stem_lower = stem.lower()
    if stem_lower.endswith(".json"):
        stem = stem[:-5]
    elif stem_lower.endswith(".parquet"):
        stem = stem[:-8]
    is_qdq = False
    if stem.endswith("_qdq"):
        stem = stem[:-4]
        is_qdq = True
    return stem, is_qdq


def _parse_file_name(file_name: str) -> tuple[str, str, str, str, int, bool]:
    stem, is_qdq = _normalize_stem(file_name)
    opset_match = re.search(r"_opset(?P<version>\d+)$", stem)
    if opset_match is None:
        raise ValueError(f"Missing _opset<version> suffix in file name: {file_name}")

    version = int(opset_match.group("version"))
    prefix = stem[: opset_match.start()]
    head, sep, domain = prefix.rpartition("_")
    if not sep:
        raise ValueError(f"Missing domain segment in file name: {file_name}")

    parts = head.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unable to parse <name>_<ep>_<device> from file name: {file_name}")

    name, ep, device = parts
    if device not in DEVICE_SET:
        raise ValueError(f"Unsupported device '{device}' in file name: {file_name}")

    return name, ep, device, domain, version, is_qdq


def _section_domain(section_name: str) -> str:
    if section_name == "ops_com_microsoft":
        return "com.microsoft"
    return "ai.onnx"


def _compute_indexes(
    providers_json_path: str,
    ops_json_path: str,
) -> IndexBundle:
    providers_json = Path(providers_json_path)
    ops_json = Path(ops_json_path)

    providers_data = _load_json_object(providers_json)
    ops_data = _load_json_object(ops_json)

    ep_device_to_key: dict[tuple[str, str], str] = {}
    key_to_ep_device: dict[str, tuple[str, str]] = {}

    for ep_name, ep_payload in providers_data.items():
        if not isinstance(ep_payload, dict):
            continue
        devices_payload = ep_payload.get("devices")
        if not isinstance(devices_payload, dict):
            continue

        for device, device_payload in devices_payload.items():
            if device not in DEVICE_SET:
                continue
            if not isinstance(device_payload, dict):
                continue

            ep_key = device_payload.get("key")
            if not isinstance(ep_key, str) or len(ep_key) != 1 or ep_key not in ALPHABET:
                raise ValueError(f"Invalid key for {ep_name}_{device}: {ep_key}")
            if ep_key in key_to_ep_device:
                raise ValueError(
                    f"Duplicate EP/device key '{ep_key}' used by "
                    f"{key_to_ep_device[ep_key]} and {(ep_name, device)}"
                )

            ep_device_to_key[(ep_name, device)] = ep_key
            key_to_ep_device[ep_key] = (ep_name, device)

    if not ep_device_to_key:
        raise ValueError(f"No EP/device key mapping found in {providers_json}")

    by_exact: dict[tuple[str, int, str], str] = {}
    by_name_version: dict[tuple[str, int], list[tuple[str, str]]] = {}
    version_key_to_entry: dict[str, tuple[str, int, str]] = {}

    for section_name in ("ops_ai_onnx", "ops_com_microsoft", "patterns", "ops"):
        section = ops_data.get(section_name)
        if not isinstance(section, dict):
            continue

        domain = _section_domain(section_name)
        for name, item in section.items():
            if not isinstance(item, dict):
                continue

            version_key_map = item.get("version_key_map")
            if not isinstance(version_key_map, dict):
                continue

            for raw_version, version_key in version_key_map.items():
                try:
                    version = int(raw_version)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid version key '{raw_version}' in {section_name}.{name}"
                    ) from exc

                if (
                    not isinstance(version_key, str)
                    or len(version_key) != 2
                    or any(ch not in ALPHABET for ch in version_key)
                ):
                    raise ValueError(
                        f"Invalid 2-char version key for {section_name}.{name} version {version}: "
                        f"{version_key}"
                    )

                if version_key in version_key_to_entry:
                    raise ValueError(
                        f"Duplicate version key '{version_key}' used by "
                        f"{version_key_to_entry[version_key]} and {(name, version, domain)}"
                    )

                by_exact[(name, version, domain)] = version_key
                by_name_version.setdefault((name, version), []).append((domain, version_key))
                version_key_to_entry[version_key] = (name, version, domain)

    if not version_key_to_entry:
        raise ValueError(f"No version key mapping found in {ops_json}")

    return ep_device_to_key, key_to_ep_device, by_exact, by_name_version, version_key_to_entry


@cache
def _get_indexes() -> IndexBundle:
    """Build (and cache) the index bundle from the bundled mapping files.

    The mapping files are fixed and shipped alongside this module; the result is
    parsed once per process and reused on every call.
    """
    providers_path, ops_path = _default_mapping_paths()
    return _compute_indexes(str(providers_path), str(ops_path))


def encode_file_name_to_4char_key(file_name: str) -> str:
    """Encode a rule file name into its 4-char case_index key prefix."""
    (
        ep_device_to_key,
        _key_to_ep_device,
        by_exact,
        by_name_version,
        _version_key_to_entry,
    ) = _get_indexes()

    name, ep, device, domain, version, is_qdq = _parse_file_name(file_name)

    ep_device_key = ep_device_to_key.get((ep, device))
    if ep_device_key is None:
        raise KeyError(f"No EP/device key found for {ep}_{device}")

    version_key = by_exact.get((name, version, domain))
    if version_key is None:
        candidates = by_name_version.get((name, version), [])
        if len(candidates) == 1:
            version_key = candidates[0][1]
        elif not candidates:
            raise KeyError(f"No version key found for {name} version {version} domain {domain}")
        else:
            text = ", ".join(
                f"{candidate_domain}:{candidate_key}"
                for candidate_domain, candidate_key in candidates
            )
            raise KeyError(
                f"Ambiguous version key for {name} version {version} domain {domain}; "
                f"candidates: {text}"
            )

    qdq_flag = "1" if is_qdq else "0"
    return f"{ep_device_key}{version_key}{qdq_flag}"


def decode_4char_key_to_folder_and_file_name(key_or_case_index: str) -> DecodedLocation:
    """Decode a 4-char key (or case_index prefix) into its folder and file name."""
    (
        _ep_device_to_key,
        key_to_ep_device,
        _by_exact,
        _by_name_version,
        version_key_to_entry,
    ) = _get_indexes()

    trimmed = key_or_case_index.strip()
    if len(trimmed) < 4:
        raise ValueError("key/case_index length must be >= 4")
    prefix = trimmed[:4]

    ep_device_key = prefix[0]
    version_key = prefix[1:3]
    qdq_flag = prefix[3]
    if ep_device_key not in ALPHABET or any(ch not in ALPHABET for ch in version_key):
        raise ValueError(f"Invalid key prefix: {prefix}")
    if qdq_flag not in {"0", "1"}:
        raise ValueError(f"Invalid qdq flag in key prefix: {prefix}")

    ep_device = key_to_ep_device.get(ep_device_key)
    if ep_device is None:
        raise KeyError(f"No EP/device mapping for key '{ep_device_key}'")
    ep, device = ep_device

    version_entry = version_key_to_entry.get(version_key)
    if version_entry is None:
        raise KeyError(f"No version mapping for key '{version_key}'")
    name, version, domain = version_entry

    folder_name = f"{ep}_{device}"
    qdq_suffix = "_qdq" if qdq_flag == "1" else ""
    file_name = f"{name}_{ep}_{device}_{domain}_opset{version}{qdq_suffix}"
    return DecodedLocation(folder_name=folder_name, file_name=file_name)


