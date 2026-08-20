# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Reranking evaluator for grouped query-document candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..utils.eval_utils import detect_reranking_dataset_mode, get_dataset_column_names
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    import torch
    from datasets import Dataset

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    text: str
    relevant: bool


@dataclass(frozen=True)
class _Group:
    group_id: str
    query: str
    candidates: tuple[_Candidate, ...]


class WinMLRerankingEvaluator(WinMLEvaluator):
    """Evaluator for cross-encoder reranking checkpoints."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from transformers import AutoTokenizer

        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "reranking"
        self._query_col = mapping.get("query_column", get_default(task, "query_column") or "input")
        self._expected_output_col = mapping.get(
            "expected_output_column",
            get_default(task, "expected_output_column") or "expected_output",
        )
        self._metadata_col = mapping.get(
            "metadata_column",
            get_default(task, "metadata_column") or "metadata",
        )
        self._candidates_col = mapping.get("candidates_column")
        self._document_col = mapping.get("document_column")
        self._group_col = mapping.get("group_column")
        self._label_col = mapping.get("label_column")
        self._candidate_id_col = mapping.get("candidate_id_column")
        self._candidate_text_key = mapping.get(
            "candidate_text_key",
            get_default(task, "candidate_text_key") or "text",
        )
        self._candidate_id_key = mapping.get(
            "candidate_id_key",
            get_default(task, "candidate_id_key") or "id",
        )
        self._metadata_group_key = mapping.get(
            "metadata_group_key",
            get_default(task, "metadata_group_key") or "query_id",
        )
        self._recall_ks = self._parse_recall_ks(
            mapping.get("recall_ks", get_default(task, "recall_ks") or "1,10")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        super().__init__(config, model)

    def prepare_pipeline(self) -> None:
        """Bypass HF pipeline postprocessing; reranking reads raw logits directly."""
        return

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No class-label alignment for grouped relevance judgments."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Score grouped candidates with raw relevance logits."""
        from .metrics import RerankingMetric

        groups = self._materialize_groups()
        metric = RerankingMetric(recall_ks=self._recall_ks)
        processed_groups = 0
        skipped_groups = 0
        processed_pairs = 0

        for group in groups:
            if not group.candidates:
                skipped_groups += 1
                continue
            scores: list[float] = []
            labels: list[bool] = []
            for candidate in group.candidates:
                scores.append(self._score_pair(group.query, candidate.text))
                labels.append(candidate.relevant)
            processed_groups += 1
            processed_pairs += len(group.candidates)
            metric.update(scores, labels)

        result = metric.compute()
        result.update(
            {
                "requested_rows": len(self.data),
                "processed_groups": processed_groups,
                "skipped_groups": skipped_groups,
                "processed_pairs": processed_pairs,
                "expanded_pairs": processed_pairs,
            }
        )
        return result

    def _materialize_groups(self) -> list[_Group]:
        column_names = set(get_dataset_column_names(self.data))
        dataset_mode = detect_reranking_dataset_mode(
            column_names,
            self.config.dataset.columns_mapping,
        )
        if dataset_mode == "pairwise":
            return self._groups_from_pairwise_rows(column_names)
        return self._groups_from_grouped_rows(column_names)

    def _groups_from_pairwise_rows(self, column_names: set[str]) -> list[_Group]:
        from ..utils.eval_utils import DatasetValidationError

        required = [self._query_col, self._document_col, self._group_col, self._label_col]
        missing = [name for name in required if name not in column_names]
        if missing:
            raise DatasetValidationError(
                f"pairwise reranking dataset is missing required column(s): {sorted(missing)}"
            )

        grouped: dict[str, list[_Candidate]] = {}
        queries: dict[str, str] = {}
        for row_index, sample in enumerate(self.data):
            group_id = str(sample[self._group_col])
            query = str(sample[self._query_col])
            document = str(sample[self._document_col])
            if not query.strip() or not document.strip():
                continue
            previous_query = queries.get(group_id)
            if previous_query is not None and previous_query != query:
                raise DatasetValidationError(
                    f"group {group_id!r} contains inconsistent query text across rows"
                )
            queries[group_id] = query
            grouped.setdefault(group_id, []).append(
                _Candidate(
                    candidate_id=str(
                        sample.get(self._candidate_id_col, f"{group_id}:{row_index}")
                        if self._candidate_id_col
                        else f"{group_id}:{row_index}"
                    ),
                    text=document,
                    relevant=self._parse_label(sample[self._label_col]),
                )
            )

        return [
            _Group(group_id=group_id, query=queries[group_id], candidates=tuple(candidates))
            for group_id, candidates in grouped.items()
        ]

    def _groups_from_grouped_rows(self, column_names: set[str]) -> list[_Group]:
        from ..utils.eval_utils import DatasetValidationError

        required = [self._query_col, self._expected_output_col, self._metadata_col]
        missing = [name for name in required if name not in column_names]
        if missing:
            raise DatasetValidationError(
                "reranking datasets require either pairwise columns "
                "(query/document/group/label) or grouped authoritative columns "
                f"({sorted(required)}); missing {sorted(missing)}"
            )

        if self._candidates_col is None:
            raise DatasetValidationError(
                "grouped reranking rows require --column candidates_column=<column> or a "
                "dataset script that materializes inline candidate passages; the authoritative "
                "MS MARCO snapshot only stores relevant passage IDs, not candidate text."
            )
        if self._candidates_col not in column_names:
            raise DatasetValidationError(
                f"grouped reranking dataset is missing candidates column {self._candidates_col!r}"
            )

        groups: list[_Group] = []
        for row_index, sample in enumerate(self.data):
            query = str(sample[self._query_col])
            if not query.strip():
                continue
            relevant_ids = set(self._parse_json_sequence(sample[self._expected_output_col]))
            metadata = self._parse_json_object(sample[self._metadata_col])
            group_id = str(metadata.get(self._metadata_group_key, row_index))
            raw_candidates = self._parse_json_sequence(sample[self._candidates_col])
            candidates: list[_Candidate] = []
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    raise DatasetValidationError(
                        f"group {group_id!r} has malformed candidate entry {candidate!r}"
                    )
                candidate_id = candidate.get(self._candidate_id_key)
                text = candidate.get(self._candidate_text_key)
                if candidate_id is None or text is None or not str(text).strip():
                    raise DatasetValidationError(
                        f"group {group_id!r} candidates must expose non-empty "
                        f"{self._candidate_id_key!r} and {self._candidate_text_key!r} fields"
                    )
                candidate_id_text = str(candidate_id)
                candidates.append(
                    _Candidate(
                        candidate_id=candidate_id_text,
                        text=str(text),
                        relevant=candidate_id_text in relevant_ids,
                    )
                )
            groups.append(_Group(group_id=group_id, query=query, candidates=tuple(candidates)))
        return groups

    def _score_pair(self, query: str, document: str) -> float:
        import torch

        tokenizer_kwargs: dict[str, Any] = {
            "truncation": True,
            "return_tensors": "pt",
        }
        max_length = self._fixed_seq_length()
        if max_length is not None:
            tokenizer_kwargs["padding"] = "max_length"
            tokenizer_kwargs["max_length"] = max_length

        encoding = self._tokenizer(query, document, **tokenizer_kwargs)
        encoding = self._pad_or_truncate(encoding, self._tokenizer)
        tensor_encoding = {
            name: value.to(self.config.pipeline_device)
            for name, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            outputs = self.model(**tensor_encoding)
        return self._extract_relevance_score(outputs)

    def _extract_relevance_score(self, outputs: Any) -> float:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        tensor = cast("torch.Tensor", logits)
        if tensor.numel() != 1:
            raise ValueError(
                "reranking expects exactly one logit per query-document pair; "
                f"got shape {tuple(tensor.shape)}"
            )
        return float(tensor.reshape(-1)[0].item())

    @staticmethod
    def _parse_label(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "positive", "relevant"}:
            return True
        if text in {"0", "false", "no", "negative", "irrelevant"}:
            return False
        raise ValueError(f"unsupported reranking relevance label: {value!r}")

    @staticmethod
    def _parse_json_sequence(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise TypeError(f"expected a JSON list, got {type(parsed).__name__}")
            return parsed
        raise TypeError(f"expected a list or JSON list string, got {type(value).__name__}")

    @staticmethod
    def _parse_json_object(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise TypeError(f"expected a JSON object, got {type(parsed).__name__}")
            return cast("dict[str, Any]", parsed)
        raise TypeError(f"expected a dict or JSON object string, got {type(value).__name__}")

    @staticmethod
    def _parse_recall_ks(raw: str) -> tuple[int, ...]:
        ks = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
        if not ks or any(k <= 0 for k in ks):
            raise ValueError(f"invalid recall_ks setting: {raw!r}")
        return ks
