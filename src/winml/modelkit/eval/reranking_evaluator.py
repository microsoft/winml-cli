# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Reranking evaluator for grouped query-document candidates."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from transformers.pipelines.text_classification import TextClassificationPipeline

from ..utils.eval_utils import detect_reranking_dataset_mode, get_dataset_column_names
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline
    from transformers.utils.generic import ModelOutput

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


def _normalize_candidate_id(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


class _RawRerankingPipeline(TextClassificationPipeline):
    """Compatibility pipeline that preserves raw model outputs.

    Reranking evaluation reads raw relevance logits directly instead of the
    text-classification pipeline's label/score postprocessing.
    """

    def postprocess(
        self,
        model_outputs: ModelOutput,
        function_to_apply: Any = None,
        top_k: Any = 1,
        _legacy: Any = True,
        **postprocess_parameters: dict[Any, Any],
    ) -> ModelOutput:
        return model_outputs


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
        self._positive_col = mapping.get("positive_column")
        self._negative_col = mapping.get("negative_column")
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
        self._max_candidates = self._parse_max_candidates(
            mapping.get("max_candidates", get_default(task, "max_candidates") or "10")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Return a Pipeline-compatible object without classification postprocessing."""
        from transformers import pipeline

        pipeline_kwargs: dict[str, Any] = {
            "device": self.config.pipeline_device,
            "pipeline_class": _RawRerankingPipeline,
            "function_to_apply": "none",
        }
        if self.config.trust_remote_code:
            pipeline_kwargs["trust_remote_code"] = True

        return cast(
            "Pipeline",
            pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.config.model_id,
                **pipeline_kwargs,
            ),
        )

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
        if dataset_mode == "grouped-text":
            return self._groups_from_text_lists()
        return self._groups_from_grouped_rows(column_names)

    def _groups_from_text_lists(self) -> list[_Group]:
        from ..utils.eval_utils import DatasetValidationError

        assert self._positive_col is not None
        assert self._negative_col is not None
        groups: list[_Group] = []
        for row_index, sample in enumerate(self.data):
            query = str(sample[self._query_col])
            if not query.strip():
                continue
            positives = self._parse_json_sequence(sample[self._positive_col])
            negatives = self._parse_json_sequence(sample[self._negative_col])
            positives = [str(text) for text in positives if str(text).strip()]
            negatives = [str(text) for text in negatives if str(text).strip()]
            if not positives:
                raise DatasetValidationError(
                    f"reranking group {row_index!r} has no non-empty positive passages"
                )

            positive_candidates = [
                _Candidate(
                    candidate_id=f"{row_index}:positive:{candidate_index}",
                    text=text,
                    relevant=True,
                )
                for candidate_index, text in enumerate(positives[: self._max_candidates])
            ]
            remaining = max(self._max_candidates - len(positive_candidates), 0)
            negative_candidates = [
                _Candidate(
                    candidate_id=f"{row_index}:negative:{candidate_index}",
                    text=text,
                    relevant=False,
                )
                for candidate_index, text in enumerate(negatives[:remaining])
            ]
            candidates: list[_Candidate] = []
            for candidate_index in range(
                max(len(positive_candidates), len(negative_candidates))
            ):
                if candidate_index < len(negative_candidates):
                    candidates.append(negative_candidates[candidate_index])
                if candidate_index < len(positive_candidates):
                    candidates.append(positive_candidates[candidate_index])
            groups.append(
                _Group(group_id=str(row_index), query=query, candidates=tuple(candidates))
            )
        return groups

    def _groups_from_pairwise_rows(self, column_names: set[str]) -> list[_Group]:
        from ..utils.eval_utils import DatasetValidationError

        required = [self._query_col, self._document_col, self._group_col, self._label_col]
        required_names = [name for name in required if name is not None]
        missing = [name for name in required_names if name not in column_names]
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
            relevant_ids = {
                _normalize_candidate_id(value)
                for value in self._parse_json_sequence(sample[self._expected_output_col])
            }
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
                candidate_id_text = _normalize_candidate_id(candidate_id)
                if not candidate_id_text:
                    raise DatasetValidationError(
                        f"group {group_id!r} candidates must expose non-empty "
                        f"{self._candidate_id_key!r} and {self._candidate_text_key!r} fields"
                    )
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

    @staticmethod
    def _parse_max_candidates(raw: str) -> int:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"invalid max_candidates setting: {raw!r}")
        return value
