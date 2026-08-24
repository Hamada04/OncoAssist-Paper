"""Primary nine-candidate inner search with content-addressed selection evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import payload_sha256
from .classifiers import LogisticRegressionConfig, fit_logistic_classifier, score_logistic_classifier
from .ctgan import AugmentedTrainingSet
from .folds import FoldProtocol, build_inner_fold_manifest, build_outer_data_fingerprint, build_outer_fold_manifest, manifest_identity_sha256, validate_inner_fold_manifest, validate_outer_fold_manifest
from .metrics import compute_binary_metrics, compute_ranking_metrics
from .protocol import ModalityAdapter, PrimaryProtocolV1, PrimarySeedManifest, PrimaryV1RunProvenance, candidate_identity_payload, candidate_identity_sha256, create_primary_v1_run_provenance, ordered_patient_ids_sha256, patient_set_sha256, validate_primary_v1_run_provenance


RATIOS = (0.25, 0.50, 0.75)
LOGISTIC_CS = (0.1, 1.0, 10.0)
_POLICY = ("mean_inner_fold_auprc", "mean_inner_fold_auroc", "lowest_inner_fold_auprc_std", "smaller_fused_latent_width", "smaller_logistic_c", "canonical_candidate_id")


@dataclass(frozen=True)
class PrimaryCandidate:
    candidate_id: str
    ae_ratio: float
    latent_dimensions: Mapping[str, int]
    fused_latent_width: int
    logistic_c: float
    augmentation: str = "minority_only_ctgan"
    class_weight: None = None

    @property
    def candidate_identity_payload(self) -> Mapping[str, Any]:
        return candidate_identity_payload(candidate_id=self.candidate_id, ae_ratio=self.ae_ratio, latent_dimensions=self.latent_dimensions, fused_latent_width=self.fused_latent_width, logistic_c=self.logistic_c, augmentation=self.augmentation, class_weight=self.class_weight)

    @property
    def candidate_identity_sha256(self) -> str:
        return candidate_identity_sha256(candidate_id=self.candidate_id, ae_ratio=self.ae_ratio, latent_dimensions=self.latent_dimensions, fused_latent_width=self.fused_latent_width, logistic_c=self.logistic_c, augmentation=self.augmentation, class_weight=self.class_weight)


@dataclass(frozen=True)
class PrimarySearchSeedBook:
    seeds: Mapping[str, int]
    seed_manifest_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.seed_manifest_identity_sha256, str) or len(self.seed_manifest_identity_sha256) != 64:
            raise ValueError("PrimarySearchSeedBook requires its PrimarySeedManifest identity.")

    def require(self, key: str) -> int:
        value = self.seeds.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Missing explicit non-negative seed: {key}")
        return value


@dataclass(frozen=True)
class OOFPrediction:
    sample_id: str
    inner_fold_id: int
    true_label: int
    decision_score: float
    uncalibrated_probability: float
    candidate_id: str
    ae_ratio: float
    logistic_c: float
    candidate_identity_sha256: str


@dataclass(frozen=True)
class SharedInnerRepresentation:
    augmented_training: AugmentedTrainingSet
    validation_fused_features: np.ndarray
    validation_sample_ids: tuple[str, ...]
    fused_feature_names: tuple[str, ...]
    validation_labels: np.ndarray
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class PrimarySharedBuilderBinding:
    """The only callable boundary accepted by the frozen Primary V1 search."""
    schema_version: str
    builder_kind: str
    protocol_id: str
    protocol_sha256: str
    outer_training_patient_set_sha256: str
    modality_adapter_sha256: str
    fold_protocol: FoldProtocol
    fold_protocol_identity_sha256: str
    seed_manifest_identity_sha256: str
    run_provenance_identity_sha256: str
    build: Callable[[float, Mapping[str, int], Mapping[str, Any], PrimarySearchSeedBook], SharedInnerRepresentation]


@dataclass(frozen=True)
class PrimaryModalitySchemaSummary:
    modality: str
    feature_count: int
    ordered_feature_schema_sha256: str
    modality_adapter_sha256: str


@dataclass(frozen=True)
class PrimaryInnerFoldSummary:
    inner_fold_id: int
    training_patient_set_sha256: str
    validation_patient_set_sha256: str
    training_ordered_patient_ids_sha256: str
    validation_ordered_patient_ids_sha256: str
    training_count: int
    validation_count: int


@dataclass(frozen=True)
class PrimaryFoldAuthority:
    """Exact deterministic outer/inner fold evidence selected for one Primary V1 search."""
    repeat_id: int
    fold_id: int
    outer_training_ids: tuple[str, ...]
    outer_testing_ids: tuple[str, ...]
    outer_training_labels: tuple[int, ...]
    inner_folds: tuple[Mapping[str, Any], ...]
    outer_manifest_identity_sha256: str
    inner_manifest_identity_sha256: str
    fold_authority_identity_sha256: str
    run_provenance_identity_sha256: str


@dataclass(frozen=True)
class PrimarySearchContext:
    """Trusted external state required to validate official Primary V1 evidence."""
    protocol: PrimaryProtocolV1
    seed_manifest: PrimarySeedManifest
    fold_protocol: FoldProtocol
    modality_adapter: ModalityAdapter
    outer_training_ids: tuple[str, ...]
    outer_training_labels: tuple[int, ...]
    outer_testing_ids: tuple[str, ...]
    inner_folds: tuple[Mapping[str, Any], ...]
    modality_schema_summaries: tuple[PrimaryModalitySchemaSummary, ...]
    inner_fold_summaries: tuple[PrimaryInnerFoldSummary, ...]
    label_by_patient: Mapping[str, int]
    validation_fold_by_patient: Mapping[str, int]
    trusted_labels_sha256: str
    repeat_id: int
    fold_id: int
    outer_manifest_identity_sha256: str
    inner_manifest_identity_sha256: str
    fold_authority_identity_sha256: str
    run_provenance_identity_sha256: str

    @property
    def outer_training_patient_set_sha256(self) -> str:
        return patient_set_sha256(self.outer_training_ids)

    @property
    def inner_fold_manifest_identity_sha256(self) -> str:
        return _fold_summary_identity(self.inner_fold_summaries)

    @property
    def identity_sha256(self) -> str:
        return payload_sha256({"protocol_sha256": self.protocol.identity_sha256, "seed_manifest_identity_sha256": self.seed_manifest.identity_sha256, "fold_protocol_identity_sha256": self.protocol.fold_protocol_identity_sha256(self.fold_protocol, self.seed_manifest, self.run_provenance_identity_sha256), "run_provenance_identity_sha256": self.run_provenance_identity_sha256, "modality_adapter_sha256": self.modality_adapter.identity_sha256, "repeat_id": self.repeat_id, "fold_id": self.fold_id, "outer_manifest_identity_sha256": self.outer_manifest_identity_sha256, "inner_manifest_identity_sha256": self.inner_manifest_identity_sha256, "fold_authority_identity_sha256": self.fold_authority_identity_sha256, "outer_training_ordered_patient_ids_sha256": ordered_patient_ids_sha256(self.outer_training_ids), "outer_testing_ordered_patient_ids_sha256": ordered_patient_ids_sha256(self.outer_testing_ids), "trusted_labels_sha256": self.trusted_labels_sha256, "validation_fold_by_patient": dict(self.validation_fold_by_patient), "inner_fold_manifest_identity_sha256": self.inner_fold_manifest_identity_sha256, "modality_schema_summaries": [_schema_payload(item) for item in self.modality_schema_summaries]})


@dataclass(frozen=True)
class RawPrimarySearchExecution:
    candidates: tuple[PrimaryCandidate, ...]
    outcomes: Mapping[str, Mapping[str, Any]]
    shared_representation_evidence: Mapping[str, Mapping[str, Any]]


class PrimarySearchIncompleteError(RuntimeError):
    """Raised when the required V1 nine-candidate grid is incomplete."""


@dataclass(frozen=True)
class PrimaryCandidateSummary:
    candidate: PrimaryCandidate
    candidate_id: str
    candidate_identity_sha256: str
    completed_inner_fold_ids: tuple[int, ...]
    inner_fold_ranking_metrics: tuple[Mapping[str, Any], ...]
    mean_inner_auprc: float
    mean_inner_auroc: float
    inner_auprc_sd: float
    identity_sha256: str


@dataclass(frozen=True)
class SelectedPrimarySearchArtifact:
    schema_version: str
    protocol_id: str
    protocol_sha256: str
    outer_training_patient_set_sha256: str
    outer_testing_patient_set_sha256: str
    repeat_id: int
    fold_id: int
    outer_manifest_identity_sha256: str
    inner_manifest_identity_sha256: str
    fold_authority_identity_sha256: str
    run_provenance_identity_sha256: str
    candidate_grid_identity_sha256: str
    selection_policy_identity_sha256: str
    inner_fold_manifest_identity_sha256: str
    modality_schema_summaries: tuple[PrimaryModalitySchemaSummary, ...]
    inner_fold_summaries: tuple[PrimaryInnerFoldSummary, ...]
    trusted_context_identity_sha256: str
    all_candidate_summaries: tuple[PrimaryCandidateSummary, ...]
    selected_candidate: PrimaryCandidate
    selected_candidate_id: str
    selected_candidate_identity_sha256: str
    selected_oof_predictions: tuple[OOFPrediction, ...]
    selected_oof_predictions_sha256: str
    search_completed_successfully: bool
    search_selection_identity_sha256: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class PrimarySearchResult:
    """Official binding emitted only after public Primary V1 search finalization."""
    context: PrimarySearchContext
    selected_search: SelectedPrimarySearchArtifact


def build_primary_candidates(feature_counts: Mapping[str, int], protocol: PrimaryProtocolV1 = PrimaryProtocolV1()) -> tuple[PrimaryCandidate, ...]:
    if set(feature_counts) != {"mGE", "mDM", "mCNA"}:
        raise ValueError("Feature counts must contain mGE, mDM, and mCNA.")
    candidates = []
    for ratio in protocol.ae_ratios:
        dimensions = {name: int(np.ceil(int(feature_counts[name]) * ratio)) for name in ("mGE", "mDM", "mCNA")}
        if any(value < 2 or value >= int(feature_counts[name]) for name, value in dimensions.items()):
            raise ValueError("Primary AE ratio is not compressive for every modality.")
        width = sum(dimensions.values())
        for c_value in protocol.logistic_cs:
            candidate_id = f"primary:ratio_{ratio:.2f}:mGE_{dimensions['mGE']}:mDM_{dimensions['mDM']}:mCNA_{dimensions['mCNA']}:width_{width}:C_{c_value:g}"
            candidates.append(PrimaryCandidate(candidate_id, ratio, dimensions, width, c_value))
    return tuple(candidates)


def _oof_payload(records: Sequence[OOFPrediction]) -> list[dict[str, Any]]:
    return [{"sample_id": item.sample_id, "inner_fold_id": item.inner_fold_id, "true_label": item.true_label, "decision_score": item.decision_score, "uncalibrated_probability": item.uncalibrated_probability, "candidate_id": item.candidate_id, "ae_ratio": item.ae_ratio, "logistic_c": item.logistic_c, "candidate_identity_sha256": item.candidate_identity_sha256} for item in records]


def _oof_hash(records: Sequence[OOFPrediction]) -> str:
    return payload_sha256(_oof_payload(tuple(sorted(records, key=lambda item: item.sample_id))))


def _summary_payload(summary: PrimaryCandidateSummary) -> dict[str, Any]:
    return {"candidate": dict(summary.candidate.candidate_identity_payload), "candidate_id": summary.candidate_id, "candidate_identity_sha256": summary.candidate_identity_sha256, "completed_inner_fold_ids": list(summary.completed_inner_fold_ids), "inner_fold_ranking_metrics": [dict(item) for item in summary.inner_fold_ranking_metrics], "mean_inner_auprc": summary.mean_inner_auprc, "mean_inner_auroc": summary.mean_inner_auroc, "inner_auprc_sd": summary.inner_auprc_sd}


def _summary_identity(summary: PrimaryCandidateSummary) -> str:
    return payload_sha256(_summary_payload(summary))


def _inner_fold_manifest_identity(inner_folds: Sequence[Mapping[str, Any]]) -> str:
    return payload_sha256([{"inner_fold_id": item["inner_fold_id"], "inner_train_ordered_patient_ids_sha256": ordered_patient_ids_sha256(item["inner_train_sample_ids"]), "inner_validation_ordered_patient_ids_sha256": ordered_patient_ids_sha256(item["inner_validation_sample_ids"])} for item in sorted(inner_folds, key=lambda item: item["inner_fold_id"])])


def _schema_payload(summary: PrimaryModalitySchemaSummary) -> dict[str, Any]:
    return {"modality": summary.modality, "feature_count": summary.feature_count, "ordered_feature_schema_sha256": summary.ordered_feature_schema_sha256, "modality_adapter_sha256": summary.modality_adapter_sha256}


def _fold_summary_payload(summary: PrimaryInnerFoldSummary) -> dict[str, Any]:
    return {"inner_fold_id": summary.inner_fold_id, "training_patient_set_sha256": summary.training_patient_set_sha256, "validation_patient_set_sha256": summary.validation_patient_set_sha256, "training_ordered_patient_ids_sha256": summary.training_ordered_patient_ids_sha256, "validation_ordered_patient_ids_sha256": summary.validation_ordered_patient_ids_sha256, "training_count": summary.training_count, "validation_count": summary.validation_count}


def _fold_summary_identity(summaries: Sequence[PrimaryInnerFoldSummary]) -> str:
    return payload_sha256([_fold_summary_payload(item) for item in sorted(summaries, key=lambda item: item.inner_fold_id)])


def _build_fold_summaries(inner_folds: Sequence[Mapping[str, Any]]) -> tuple[PrimaryInnerFoldSummary, ...]:
    return tuple(PrimaryInnerFoldSummary(item["inner_fold_id"], patient_set_sha256(item["inner_train_sample_ids"]), patient_set_sha256(item["inner_validation_sample_ids"]), ordered_patient_ids_sha256(item["inner_train_sample_ids"]), ordered_patient_ids_sha256(item["inner_validation_sample_ids"]), len(item["inner_train_sample_ids"]), len(item["inner_validation_sample_ids"])) for item in sorted(inner_folds, key=lambda item: item["inner_fold_id"]))


def _fold_authority_identity(protocol: PrimaryProtocolV1, seed_manifest: PrimarySeedManifest, fold_protocol: FoldProtocol, outer_manifest_identity: str, inner_manifest_identity: str, repeat_id: int, fold_id: int, provenance_identity: str) -> str:
    return payload_sha256({"schema_version": "primary-fold-authority-v1", "protocol_id": protocol.protocol_id, "protocol_sha256": protocol.identity_sha256, "run_provenance_identity_sha256": provenance_identity, "seed_manifest_identity_sha256": seed_manifest.identity_sha256, "fold_protocol_identity_sha256": protocol.fold_protocol_identity_sha256(fold_protocol, seed_manifest, provenance_identity), "outer_manifest_identity_sha256": outer_manifest_identity, "inner_manifest_identity_sha256": inner_manifest_identity, "repeat_id": repeat_id, "fold_id": fold_id})


def _derive_primary_fold_authority(protocol: PrimaryProtocolV1, provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any], repeat_id: int, fold_id: int) -> PrimaryFoldAuthority:
    """Rebuild and select the only Primary V1 fold membership accepted publicly."""
    if not isinstance(protocol, PrimaryProtocolV1) or not isinstance(aligned_data, Mapping):
        raise TypeError("Primary fold authority requires protocol and canonical aligned data.")
    validate_primary_v1_run_provenance(provenance, protocol=protocol, aligned_data=aligned_data)
    if type(repeat_id) is not int or type(fold_id) is not int or repeat_id not in range(protocol.outer_n_repeats) or fold_id not in range(protocol.outer_n_splits):
        raise ValueError("Primary fold authority repeat_id/fold_id is outside the frozen protocol range.")
    try:
        sample_ids = tuple(str(value) for value in aligned_data["sample_ids"])
        labels = np.asarray(aligned_data["y_binary"], dtype=int)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Primary fold authority requires canonical aligned sample_ids and y_binary labels.") from error
    adapter = ModalityAdapter.from_aligned_data(aligned_data)
    seed_manifest = PrimarySeedManifest.generate_primary(provenance, build_primary_candidates({modality: len(aligned_data["feature_columns"][adapter.bindings[modality].feature_key]) for modality in ("mGE", "mDM", "mCNA")}, protocol))
    fold_protocol = protocol.make_fold_protocol(seed_manifest, provenance)
    data_fingerprint = build_outer_data_fingerprint(aligned_data)
    outer_manifest = build_outer_fold_manifest(sample_ids, labels, data_fingerprint, fold_protocol)
    validate_outer_fold_manifest(outer_manifest, sample_ids, labels, data_fingerprint, fold_protocol)
    outer_identity = manifest_identity_sha256(outer_manifest)
    matching_outer = [record for record in outer_manifest["folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id)]
    if len(matching_outer) != 1:
        raise ValueError("Primary fold authority requires exactly one canonical outer fold record.")
    inner_manifest = build_inner_fold_manifest(outer_manifest, outer_identity, sample_ids, labels, data_fingerprint, fold_protocol)
    validate_inner_fold_manifest(inner_manifest, outer_manifest, outer_identity, sample_ids, labels, data_fingerprint, fold_protocol)
    inner_identity = manifest_identity_sha256(inner_manifest)
    matching_inner = [record for record in inner_manifest["outer_folds"] if (record["repeat_id"], record["fold_id"]) == (repeat_id, fold_id)]
    if len(matching_inner) != 1:
        raise ValueError("Primary fold authority requires exactly one canonical inner fold record.")
    outer_record, inner_record = matching_outer[0], matching_inner[0]
    label_by_id = dict(zip(sample_ids, labels.tolist()))
    outer_training_ids = tuple(outer_record["train_sample_ids"])
    outer_testing_ids = tuple(outer_record["test_sample_ids"])
    inner_folds = tuple({"inner_fold_id": record["inner_fold_id"], "inner_train_sample_ids": tuple(record["inner_train_sample_ids"]), "inner_validation_sample_ids": tuple(record["inner_validation_sample_ids"])} for record in sorted(inner_record["inner_folds"], key=lambda record: record["inner_fold_id"]))
    return PrimaryFoldAuthority(repeat_id, fold_id, outer_training_ids, outer_testing_ids, tuple(label_by_id[item] for item in outer_training_ids), inner_folds, outer_identity, inner_identity, _fold_authority_identity(protocol, seed_manifest, fold_protocol, outer_identity, inner_identity, repeat_id, fold_id, provenance.identity_sha256), provenance.identity_sha256)


def _build_primary_search_context(protocol: PrimaryProtocolV1, seed_manifest: PrimarySeedManifest, fold_protocol: FoldProtocol, modality_adapter: ModalityAdapter, aligned_data: Mapping[str, Any], fold_authority: PrimaryFoldAuthority, provenance: PrimaryV1RunProvenance) -> PrimarySearchContext:
    """Canonicalize trusted Primary V1 cohort, fold, and modality schema context."""
    if not isinstance(protocol, PrimaryProtocolV1) or not isinstance(modality_adapter, ModalityAdapter) or not isinstance(aligned_data, Mapping):
        raise TypeError("Primary search context requires frozen protocol, adapter, and aligned data.")
    validate_primary_v1_run_provenance(provenance, protocol=protocol, aligned_data=aligned_data)
    protocol.validate_primary_seed_manifest(seed_manifest, provenance)
    if fold_protocol != protocol.make_fold_protocol(seed_manifest, provenance):
        raise ValueError("Primary search context FoldProtocol is not bound to its seed manifest.")
    if not isinstance(fold_authority, PrimaryFoldAuthority):
        raise TypeError("Primary search context requires canonical PrimaryFoldAuthority.")
    if fold_authority.run_provenance_identity_sha256 != provenance.identity_sha256 or fold_authority.fold_authority_identity_sha256 != _fold_authority_identity(protocol, seed_manifest, fold_protocol, fold_authority.outer_manifest_identity_sha256, fold_authority.inner_manifest_identity_sha256, fold_authority.repeat_id, fold_authority.fold_id, provenance.identity_sha256):
        raise ValueError("Primary search context fold authority identity differs.")
    ids, labels = fold_authority.outer_training_ids, tuple(int(item) for item in fold_authority.outer_training_labels)
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("Primary search context requires unique binary outer-training records.")
    testing_ids = tuple(str(item) for item in fold_authority.outer_testing_ids)
    if not testing_ids or len(testing_ids) != len(set(testing_ids)) or set(testing_ids).intersection(ids):
        raise ValueError("Primary search context outer-test authority is invalid.")
    _validate_inner_folds(fold_authority.inner_folds, ids)
    normalized_folds = tuple({"inner_fold_id": item["inner_fold_id"], "inner_train_sample_ids": tuple(str(value) for value in item["inner_train_sample_ids"]), "inner_validation_sample_ids": tuple(str(value) for value in item["inner_validation_sample_ids"])} for item in sorted(fold_authority.inner_folds, key=lambda item: item["inner_fold_id"]))
    summaries = []
    for modality in ("mGE", "mDM", "mCNA"):
        binding = modality_adapter.bindings[modality]
        names = tuple(aligned_data.get("feature_columns", {}).get(binding.feature_key, ()))
        frame = aligned_data.get(binding.matrix_key)
        if not names or frame is None or tuple(frame.columns) != names or payload_sha256(list(names)) != binding.source_schema_sha256:
            raise ValueError("Primary search modality schema differs from its adapter.")
        summaries.append(PrimaryModalitySchemaSummary(modality, len(names), binding.source_schema_sha256, modality_adapter.identity_sha256))
    label_by_patient = MappingProxyType(dict(zip(ids, labels)))
    validation_fold_by_patient = MappingProxyType({patient_id: fold["inner_fold_id"] for fold in normalized_folds for patient_id in fold["inner_validation_sample_ids"]})
    if set(validation_fold_by_patient) != set(ids):
        raise ValueError("Primary search context validation-fold coverage is incomplete.")
    label_identity = payload_sha256([{"sample_id": patient_id, "true_label": label_by_patient[patient_id]} for patient_id in ids])
    return PrimarySearchContext(protocol, seed_manifest, fold_protocol, modality_adapter, ids, labels, testing_ids, normalized_folds, tuple(summaries), _build_fold_summaries(normalized_folds), label_by_patient, validation_fold_by_patient, label_identity, fold_authority.repeat_id, fold_authority.fold_id, fold_authority.outer_manifest_identity_sha256, fold_authority.inner_manifest_identity_sha256, fold_authority.fold_authority_identity_sha256, provenance.identity_sha256)


def _selection_key(summary: PrimaryCandidateSummary) -> tuple[Any, ...]:
    candidate = summary.candidate
    return (-summary.mean_inner_auprc, -summary.mean_inner_auroc, summary.inner_auprc_sd, candidate.fused_latent_width, candidate.logistic_c, candidate.candidate_id)


def _selection_identity(artifact: SelectedPrimarySearchArtifact) -> str:
    return payload_sha256({"schema_version": "primary-search-selection-v1", "protocol_sha256": artifact.protocol_sha256, "run_provenance_identity_sha256": artifact.run_provenance_identity_sha256, "outer_training_patient_set_sha256": artifact.outer_training_patient_set_sha256, "outer_testing_patient_set_sha256": artifact.outer_testing_patient_set_sha256, "repeat_id": artifact.repeat_id, "fold_id": artifact.fold_id, "outer_manifest_identity_sha256": artifact.outer_manifest_identity_sha256, "inner_manifest_identity_sha256": artifact.inner_manifest_identity_sha256, "fold_authority_identity_sha256": artifact.fold_authority_identity_sha256, "candidate_grid_identity_sha256": artifact.candidate_grid_identity_sha256, "selection_policy_identity_sha256": artifact.selection_policy_identity_sha256, "inner_fold_manifest_identity_sha256": artifact.inner_fold_manifest_identity_sha256, "modality_schema_summaries": [_schema_payload(item) for item in artifact.modality_schema_summaries], "inner_fold_summaries": [_fold_summary_payload(item) for item in artifact.inner_fold_summaries], "trusted_context_identity_sha256": artifact.trusted_context_identity_sha256, "all_candidate_summary_identities": [summary.identity_sha256 for summary in sorted(artifact.all_candidate_summaries, key=lambda item: item.candidate_id)], "selected_candidate_identity_sha256": artifact.selected_candidate_identity_sha256, "selected_oof_predictions_sha256": artifact.selected_oof_predictions_sha256, "search_completed_successfully": artifact.search_completed_successfully})


def _validate_inner_folds(inner_folds: Sequence[Mapping[str, Any]], outer_ids: tuple[str, ...]) -> str:
    if not isinstance(inner_folds, Sequence) or len(inner_folds) != 3:
        raise ValueError("Primary search requires exactly three frozen inner folds.")
    outer_set, seen, validation_coverage = set(outer_ids), set(), []
    normalized = []
    for inner in inner_folds:
        if not isinstance(inner, Mapping) or type(inner.get("inner_fold_id")) is not int:
            raise ValueError("Primary inner folds require integer fold IDs.")
        fold_id = inner["inner_fold_id"]
        if fold_id not in {0, 1, 2} or fold_id in seen:
            raise ValueError("Primary inner folds must contain each exact fold ID once.")
        seen.add(fold_id)
        train = tuple(str(item) for item in inner.get("inner_train_sample_ids", ()))
        validation = tuple(str(item) for item in inner.get("inner_validation_sample_ids", ()))
        if not train or not validation or len(train) != len(set(train)) or len(validation) != len(set(validation)):
            raise ValueError("Primary inner folds require non-empty unique train and validation IDs.")
        train_set, validation_set = set(train), set(validation)
        if train_set & validation_set or train_set | validation_set != outer_set:
            raise ValueError("Primary inner fold must partition the complete bound outer-training cohort.")
        validation_coverage.extend(validation)
        normalized.append({"inner_fold_id": fold_id, "inner_train_sample_ids": train, "inner_validation_sample_ids": validation})
    if seen != {0, 1, 2} or len(validation_coverage) != len(outer_ids) or set(validation_coverage) != outer_set or len(set(validation_coverage)) != len(outer_ids):
        raise ValueError("Primary inner validation folds must cover the bound cohort exactly once.")
    return _inner_fold_manifest_identity(normalized)


def _validate_binding(binding: PrimarySharedBuilderBinding, protocol: PrimaryProtocolV1, outer_ids: tuple[str, ...], seed_book: PrimarySearchSeedBook, run_provenance_identity_sha256: str) -> None:
    if not isinstance(binding, PrimarySharedBuilderBinding) or binding.schema_version != "primary-shared-builder-v1" or binding.builder_kind != "primary-inner-v1" or not callable(binding.build):
        raise TypeError("Primary V1 search requires a canonical PrimarySharedBuilderBinding, not a callback.")
    if binding.protocol_id != protocol.protocol_id or binding.protocol_sha256 != protocol.identity_sha256:
        raise ValueError("Primary shared builder protocol binding differs.")
    if binding.outer_training_patient_set_sha256 != patient_set_sha256(outer_ids):
        raise ValueError("Primary shared builder outer-training cohort differs.")
    if not isinstance(binding.modality_adapter_sha256, str) or len(binding.modality_adapter_sha256) != 64:
        raise ValueError("Primary shared builder modality adapter identity is invalid.")
    if binding.seed_manifest_identity_sha256 != seed_book.seed_manifest_identity_sha256:
        raise ValueError("Primary shared builder seed manifest differs.")
    if binding.run_provenance_identity_sha256 != run_provenance_identity_sha256:
        raise ValueError("Primary shared builder run provenance differs.")
    protocol.validate_fold_protocol(binding.fold_protocol)
    expected_fold_protocol = FoldProtocol(5, 5, seed_book.require("outer:fold_generation"), 3, seed_book.require("inner:fold_generation"))
    if binding.fold_protocol != expected_fold_protocol:
        raise ValueError("Primary shared builder FoldProtocol does not use its seed-manifest fold seeds.")
    expected_fold_identity = protocol.fold_protocol_identity_sha256(binding.fold_protocol, binding.seed_manifest_identity_sha256, run_provenance_identity_sha256)
    if binding.fold_protocol_identity_sha256 != expected_fold_identity:
        raise ValueError("Primary shared builder fold-protocol binding differs.")


def validate_selected_primary_search_artifact(artifact: SelectedPrimarySearchArtifact, context: PrimarySearchContext) -> None:
    """Validate content-addressed selection integrity; it does not attest function origin."""
    if not isinstance(context, PrimarySearchContext) or not isinstance(artifact, SelectedPrimarySearchArtifact) or artifact.schema_version != "selected-primary-search-artifact-v1":
        raise TypeError("A SelectedPrimarySearchArtifact V1 payload is required.")
    protocol = context.protocol
    if artifact.protocol_id != protocol.protocol_id or artifact.protocol_sha256 != protocol.identity_sha256:
        raise ValueError("Selected search artifact protocol binding differs.")
    if artifact.outer_training_patient_set_sha256 != context.outer_training_patient_set_sha256 or artifact.outer_testing_patient_set_sha256 != patient_set_sha256(context.outer_testing_ids) or artifact.trusted_context_identity_sha256 != context.identity_sha256:
        raise ValueError("Selected search artifact outer-training cohort differs.")
    if artifact.run_provenance_identity_sha256 != context.run_provenance_identity_sha256 or artifact.repeat_id != context.repeat_id or artifact.fold_id != context.fold_id or artifact.outer_manifest_identity_sha256 != context.outer_manifest_identity_sha256 or artifact.inner_manifest_identity_sha256 != context.inner_manifest_identity_sha256 or artifact.fold_authority_identity_sha256 != context.fold_authority_identity_sha256:
        raise ValueError("Selected search artifact authoritative fold binding differs.")
    if artifact.modality_schema_summaries != context.modality_schema_summaries or artifact.inner_fold_summaries != context.inner_fold_summaries or artifact.inner_fold_manifest_identity_sha256 != context.inner_fold_manifest_identity_sha256:
        raise ValueError("Selected search artifact structural context differs.")
    summaries = tuple(artifact.all_candidate_summaries)
    if len(summaries) != 9 or len({item.candidate_id for item in summaries}) != 9:
        raise ValueError("Selected search artifact requires exactly nine unique candidate summaries.")
    expected_candidates = build_primary_candidates({item.modality: item.feature_count for item in context.modality_schema_summaries}, protocol)
    expected_by_pair = {(item.ae_ratio, item.logistic_c): item for item in expected_candidates}
    expected_pairs = set(expected_by_pair)
    pairs = set()
    for summary in summaries:
        if not isinstance(summary, PrimaryCandidateSummary) or not isinstance(summary.candidate, PrimaryCandidate):
            raise ValueError("Selected search artifact contains an invalid candidate summary.")
        candidate = summary.candidate
        if summary.candidate_id != candidate.candidate_id or summary.candidate_identity_sha256 != candidate.candidate_identity_sha256 or summary.identity_sha256 != _summary_identity(summary):
            raise ValueError("Selected search artifact candidate summary identity differs.")
        if candidate != expected_by_pair.get((candidate.ae_ratio, candidate.logistic_c)):
            raise ValueError("Selected search artifact candidate is outside Primary V1.")
        pairs.add((candidate.ae_ratio, candidate.logistic_c))
        metrics = tuple(summary.inner_fold_ranking_metrics)
        if tuple(summary.completed_inner_fold_ids) != (0, 1, 2) or len(metrics) != 3 or {item.get("inner_fold_id") for item in metrics if isinstance(item, Mapping)} != {0, 1, 2}:
            raise ValueError("Each primary candidate summary requires exact three-fold completion.")
        try:
            auprcs = np.asarray([float(item["auprc"]) for item in sorted(metrics, key=lambda item: item["inner_fold_id"])])
            aurocs = np.asarray([float(item["auroc"]) for item in sorted(metrics, key=lambda item: item["inner_fold_id"])])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Candidate summary ranking metrics are invalid.") from error
        if not np.isfinite(auprcs).all() or not np.isfinite(aurocs).all() or np.any((auprcs < 0) | (auprcs > 1)) or np.any((aurocs < 0) | (aurocs > 1)) or not all(isfinite(value) for value in (summary.mean_inner_auprc, summary.mean_inner_auroc, summary.inner_auprc_sd)):
            raise ValueError("Candidate summary metrics must be finite probabilities.")
        if (summary.mean_inner_auprc, summary.mean_inner_auroc, summary.inner_auprc_sd) != (float(np.mean(auprcs)), float(np.mean(aurocs)), float(np.std(auprcs, ddof=0))):
            raise ValueError("Candidate summary aggregate metrics do not recompute.")
    if pairs != expected_pairs:
        raise ValueError("Selected search artifact candidate grid differs from Primary V1.")
    expected_grid = payload_sha256([item.candidate_identity_sha256 for item in sorted(summaries, key=lambda item: item.candidate_id)])
    if artifact.candidate_grid_identity_sha256 != expected_grid or artifact.selection_policy_identity_sha256 != payload_sha256(list(_POLICY)):
        raise ValueError("Selected search artifact grid or policy identity differs.")
    winner = min(summaries, key=_selection_key)
    if artifact.selected_candidate != winner.candidate or artifact.selected_candidate_id != winner.candidate_id or artifact.selected_candidate_identity_sha256 != winner.candidate_identity_sha256:
        raise ValueError("Selected search artifact winner does not recompute from complete grid evidence.")
    records = tuple(artifact.selected_oof_predictions)
    if not records or records != tuple(sorted(records, key=lambda item: item.sample_id)):
        raise ValueError("Selected OOF records must be non-empty and canonically ordered.")
    ids = tuple(record.sample_id for record in records)
    if len(ids) != len(set(ids)) or any(not isinstance(record, OOFPrediction) or not record.sample_id.strip() or record.sample_id.startswith("SYNTHETIC:") or record.candidate_id != winner.candidate_id or record.candidate_identity_sha256 != winner.candidate_identity_sha256 or record.sample_id not in context.label_by_patient or record.true_label != context.label_by_patient[record.sample_id] or record.inner_fold_id != context.validation_fold_by_patient[record.sample_id] or not np.isfinite(record.decision_score) or not np.isfinite(record.uncalibrated_probability) or not 0 <= record.uncalibrated_probability <= 1 for record in records):
        raise ValueError("Selected OOF records are not valid selected-candidate real-patient evidence.")
    if patient_set_sha256(ids) != artifact.outer_training_patient_set_sha256 or set(ids) != set(context.outer_training_ids) or _oof_hash(records) != artifact.selected_oof_predictions_sha256:
        raise ValueError("Selected OOF membership or content identity differs.")
    if artifact.search_completed_successfully is not True or artifact.search_selection_identity_sha256 != _selection_identity(artifact):
        raise ValueError("Selected search artifact completion or selection identity differs.")


def validate_primary_search_result(result: PrimarySearchResult, *, run_provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any]) -> None:
    if not isinstance(result, PrimarySearchResult) or not isinstance(result.context, PrimarySearchContext):
        raise TypeError("PrimarySearchResult with trusted context is required.")
    context = result.context
    protocol = PrimaryProtocolV1()
    validate_primary_v1_run_provenance(run_provenance, protocol=protocol, aligned_data=aligned_data)
    if context.run_provenance_identity_sha256 != run_provenance.identity_sha256:
        raise ValueError("Primary search result run provenance differs.")
    adapter = ModalityAdapter.from_aligned_data(aligned_data)
    candidates = build_primary_candidates({modality: len(aligned_data["feature_columns"][adapter.bindings[modality].feature_key]) for modality in ("mGE", "mDM", "mCNA")}, protocol)
    expected_manifest = PrimarySeedManifest.generate_primary(run_provenance, candidates)
    expected_fold_protocol = protocol.make_fold_protocol(expected_manifest, run_provenance)
    expected_authority = _derive_primary_fold_authority(protocol, run_provenance, aligned_data, context.repeat_id, context.fold_id)
    expected_context = _build_primary_search_context(protocol, expected_manifest, expected_fold_protocol, adapter, aligned_data, expected_authority, run_provenance)
    if context != expected_context:
        raise ValueError("Primary search result context does not match canonical run reconstruction.")
    validate_selected_primary_search_artifact(result.selected_search, expected_context)


def _run_primary_inner_search_with_builder(context: PrimarySearchContext, shared_builder: PrimarySharedBuilderBinding, classifier_fitter: Callable[[AugmentedTrainingSet, LogisticRegressionConfig, int], Any] = fit_logistic_classifier, classifier_scorer: Callable[[Any, np.ndarray, Sequence[str], Sequence[str]], Any] = score_logistic_classifier) -> Mapping[str, Any]:
    """Internal execution seam. Official callers use run_primary_inner_search."""
    if not isinstance(context, PrimarySearchContext):
        raise TypeError("Internal search execution requires PrimarySearchContext.")
    ids, labels, protocol = context.outer_training_ids, np.asarray(context.outer_training_labels, dtype=int), context.protocol
    inner_folds, seed_book = context.inner_folds, context.seed_manifest.primary_search_seed_book()
    _validate_binding(shared_builder, protocol, ids, seed_book, context.run_provenance_identity_sha256)
    inner_manifest_identity = _validate_inner_folds(inner_folds, ids)
    label_by_id, candidates = dict(zip(ids, labels.tolist())), build_primary_candidates({item.modality: item.feature_count for item in context.modality_schema_summaries}, protocol)
    shared_evidence: dict[str, Mapping[str, Any]] = {}
    outcomes = {candidate.candidate_id: {"candidate": candidate, "folds": [], "oof": [], "failures": []} for candidate in candidates}
    for ratio in RATIOS:
        ratio_candidates = [item for item in candidates if item.ae_ratio == ratio]
        dimensions = ratio_candidates[0].latent_dimensions
        for inner in inner_folds:
            inner_id = inner["inner_fold_id"]
            validation_ids = tuple(str(item) for item in inner["inner_validation_sample_ids"])
            failure_context = {"inner_fold_id": inner_id, "train_ids_ordered_patient_ids_sha256": ordered_patient_ids_sha256(inner["inner_train_sample_ids"]), "validation_ids_ordered_patient_ids_sha256": ordered_patient_ids_sha256(validation_ids), "ratio": ratio}
            try:
                for key in (f"ae_split:{inner_id}:{ratio}", f"ctgan:{inner_id}:{ratio}"):
                    seed_book.require(key)
                for modality in ("mGE", "mDM", "mCNA"):
                    seed_book.require(f"ae_select:{inner_id}:{ratio}:{modality}")
                    seed_book.require(f"ae_refit:{inner_id}:{ratio}:{modality}")
                shared = shared_builder.build(ratio, dimensions, inner, seed_book)
                if not isinstance(shared, SharedInnerRepresentation) or tuple(shared.validation_sample_ids) != validation_ids or not np.array_equal(np.asarray(shared.validation_labels, dtype=int), np.asarray([label_by_id[item] for item in validation_ids], dtype=int)):
                    raise ValueError("Shared representation validation data does not match frozen inner fold.")
                augmentation, validation_features, feature_names, returned_labels = shared.augmented_training, shared.validation_fused_features, shared.fused_feature_names, shared.validation_labels
                shared_evidence[f"inner_{inner_id}:ratio_{ratio}"] = MappingProxyType(dict(shared.evidence))
            except Exception as error:
                for candidate in ratio_candidates:
                    outcomes[candidate.candidate_id]["failures"].append({"candidate_id": candidate.candidate_id, "inner_fold_id": inner_id, "component": "shared_representation_or_ctgan", "exception_type": type(error).__name__, "message": str(error), "context": failure_context})
                continue
            for candidate in ratio_candidates:
                try:
                    fitted = classifier_fitter(augmentation, LogisticRegressionConfig(candidate.logistic_c), seed_book.require(f"lr:{inner_id}:{ratio}:{candidate.logistic_c}"))
                    scores = classifier_scorer(fitted, validation_features, validation_ids, feature_names)
                    decision, probability, true = np.asarray(scores.decision_scores, dtype=float), np.asarray(scores.probabilities, dtype=float), np.asarray(returned_labels, dtype=int)
                    ranking = compute_ranking_metrics(true, decision)
                    diagnostic = compute_binary_metrics(true, probability, 0.5, "probability")
                    records = [OOFPrediction(sample_id, inner_id, int(label), float(raw), float(prob), candidate.candidate_id, ratio, candidate.logistic_c, candidate.candidate_identity_sha256) for sample_id, label, raw, prob in zip(validation_ids, true, decision, probability)]
                    outcomes[candidate.candidate_id]["folds"].append({"inner_fold_id": inner_id, "ranking": ranking, "diagnostic_uncalibrated_threshold_metrics": diagnostic})
                    outcomes[candidate.candidate_id]["oof"].extend(records)
                except Exception as error:
                    outcomes[candidate.candidate_id]["failures"].append({"candidate_id": candidate.candidate_id, "inner_fold_id": inner_id, "component": "classifier_or_metrics", "exception_type": type(error).__name__, "message": str(error), "context": failure_context})
    return RawPrimarySearchExecution(tuple(candidates), MappingProxyType({key: MappingProxyType(dict(value)) for key, value in outcomes.items()}), MappingProxyType(dict(shared_evidence)))


def _finalize_primary_search_execution(context: PrimarySearchContext, execution: RawPrimarySearchExecution) -> PrimarySearchResult:
    """Only the public wrapper turns complete raw execution into official evidence."""
    ids, protocol, candidates = context.outer_training_ids, context.protocol, execution.candidates
    outcomes, shared_evidence = execution.outcomes, execution.shared_representation_evidence
    summaries, raw_summaries = [], []
    for result in outcomes.values():
        valid = not result["failures"] and len(result["folds"]) == 3 and len(result["oof"]) == len(ids) and {record.sample_id for record in result["oof"]} == set(ids) and len({record.sample_id for record in result["oof"]}) == len(ids)
        raw = {**result, "valid": valid}
        if valid:
            folds = sorted(result["folds"], key=lambda item: item["inner_fold_id"])
            metrics = tuple({"inner_fold_id": item["inner_fold_id"], "auprc": float(item["ranking"].auprc), "auroc": float(item["ranking"].auroc)} for item in folds)
            auprcs, aurocs = np.asarray([item["auprc"] for item in metrics]), np.asarray([item["auroc"] for item in metrics])
            draft = PrimaryCandidateSummary(result["candidate"], result["candidate"].candidate_id, result["candidate"].candidate_identity_sha256, (0, 1, 2), metrics, float(np.mean(auprcs)), float(np.mean(aurocs)), float(np.std(auprcs, ddof=0)), "")
            summary = replace(draft, identity_sha256=_summary_identity(draft))
            summaries.append(summary)
            raw.update({"mean_auprc": summary.mean_inner_auprc, "mean_auroc": summary.mean_inner_auroc, "std_auprc": summary.inner_auprc_sd, "pooled_oof": compute_ranking_metrics([record.true_label for record in result["oof"]], [record.decision_score for record in result["oof"]])})
        raw_summaries.append(raw)
    failures = [failure for result in outcomes.values() for failure in result["failures"]]
    if failures or len(summaries) != 9:
        raise PrimarySearchIncompleteError(f"Primary V1 search grid is incomplete: {failures}")
    selected = min(summaries, key=_selection_key)
    selected_records = tuple(sorted(outcomes[selected.candidate_id]["oof"], key=lambda item: item.sample_id))
    artifact_draft = SelectedPrimarySearchArtifact("selected-primary-search-artifact-v1", protocol.protocol_id, protocol.identity_sha256, patient_set_sha256(ids), patient_set_sha256(context.outer_testing_ids), context.repeat_id, context.fold_id, context.outer_manifest_identity_sha256, context.inner_manifest_identity_sha256, context.fold_authority_identity_sha256, context.run_provenance_identity_sha256, payload_sha256([item.candidate_identity_sha256 for item in sorted(summaries, key=lambda item: item.candidate_id)]), payload_sha256(list(_POLICY)), context.inner_fold_manifest_identity_sha256, context.modality_schema_summaries, context.inner_fold_summaries, context.identity_sha256, tuple(sorted(summaries, key=lambda item: item.candidate_id)), selected.candidate, selected.candidate_id, selected.candidate_identity_sha256, selected_records, _oof_hash(selected_records), True, "", {"run_provenance_identity_sha256": context.run_provenance_identity_sha256, "outer_training_ordered_patient_ids_sha256": ordered_patient_ids_sha256(ids), "outer_testing_ordered_patient_ids_sha256": ordered_patient_ids_sha256(context.outer_testing_ids), "seed_manifest_identity_sha256": context.seed_manifest.identity_sha256, "fold_protocol_identity_sha256": context.protocol.fold_protocol_identity_sha256(context.fold_protocol, context.seed_manifest, context.run_provenance_identity_sha256), "outer_manifest_identity_sha256": context.outer_manifest_identity_sha256, "inner_manifest_identity_sha256": context.inner_manifest_identity_sha256, "fold_authority_identity_sha256": context.fold_authority_identity_sha256, "modality_adapter_sha256": context.modality_adapter.identity_sha256})
    artifact = replace(artifact_draft, search_selection_identity_sha256=_selection_identity(artifact_draft))
    validate_selected_primary_search_artifact(artifact, context)
    official = PrimarySearchResult(context, artifact)
    return official


def run_primary_inner_search(*, run_provenance: PrimaryV1RunProvenance, aligned_data: Mapping[str, Any], repeat_id: int, fold_id: int, ae_training_config: Any, ctgan_config: Any, ae_validation_fraction: float, synthetic_namespace_prefix: str) -> PrimarySearchResult:
    """Official Primary V1 search. It owns construction of the production builder."""
    protocol = PrimaryProtocolV1()
    validate_primary_v1_run_provenance(run_provenance, protocol=protocol, aligned_data=aligned_data)
    modality_adapter = ModalityAdapter.from_aligned_data(aligned_data)
    candidates = build_primary_candidates({modality: len(aligned_data["feature_columns"][modality_adapter.bindings[modality].feature_key]) for modality in ("mGE", "mDM", "mCNA")}, protocol)
    seed_manifest = PrimarySeedManifest.generate_primary(run_provenance, candidates)
    fold_protocol = protocol.make_fold_protocol(seed_manifest, run_provenance)
    fold_authority = _derive_primary_fold_authority(protocol, run_provenance, aligned_data, repeat_id, fold_id)
    context = _build_primary_search_context(protocol, seed_manifest, fold_protocol, modality_adapter, aligned_data, fold_authority, run_provenance)
    from .primary_inner import make_primary_inner_builder
    outer_fold_identity = {"schema_version": "primary-fold-authority-v1", "repeat_id": context.repeat_id, "fold_id": context.fold_id, "outer_manifest_identity_sha256": context.outer_manifest_identity_sha256, "inner_manifest_identity_sha256": context.inner_manifest_identity_sha256, "fold_authority_identity_sha256": context.fold_authority_identity_sha256, "outer_training_ordered_patient_ids_sha256": ordered_patient_ids_sha256(context.outer_training_ids), "outer_testing_ordered_patient_ids_sha256": ordered_patient_ids_sha256(context.outer_testing_ids)}
    builder = make_primary_inner_builder(aligned_data, context.modality_adapter, context.outer_training_ids, context.outer_training_labels, ae_training_config, ctgan_config, protocol=context.protocol, seed_manifest=context.seed_manifest, run_provenance=run_provenance, fold_protocol=context.fold_protocol, ae_validation_fraction=ae_validation_fraction, synthetic_namespace_prefix=synthetic_namespace_prefix, protocol_hash=context.protocol.identity_sha256, outer_fold_identity=outer_fold_identity)
    result = _finalize_primary_search_execution(context, _run_primary_inner_search_with_builder(context, builder))
    validate_primary_search_result(result, run_provenance=run_provenance, aligned_data=aligned_data)
    return result
