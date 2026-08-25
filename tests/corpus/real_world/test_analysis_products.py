from __future__ import annotations

import pytest

from kinematic_classifier_sandbox.corpus.real_world.analysis_products import (
    AnalysisProductId,
    AnalysisProductManifest,
    build_analysis_product_manifest,
    write_analysis_product_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import SourceEvidenceState

from .test_portfolio import _episode, _prepared_registry, _snapshot


def test_source_and_kinematic_products_exclude_classifier_assets(tmp_path) -> None:
    episode = _episode()
    snapshot = _snapshot(tmp_path, (episode,))
    registry = _prepared_registry()

    source_audit = build_analysis_product_manifest(
        snapshot,
        (episode,),
        registry,
        product_id=AnalysisProductId.SOURCE_AUDIT,
    )
    kinematic = build_analysis_product_manifest(
        snapshot,
        (episode,),
        registry,
        product_id=AnalysisProductId.KINEMATIC_ANALYSIS,
    )

    assert source_audit.episodes[0].classifier_asset is None
    assert source_audit.episodes[0].state_view_ids == ("land-episode-1-observed",)
    assert kinematic.episodes[0].classifier_asset is None
    assert kinematic.episodes[0].state_assets[0].path == "states/land-episode-1.json"


def test_classifier_ladder_manifest_is_asset_only_and_round_trips(tmp_path) -> None:
    episode = _episode()
    snapshot = _snapshot(tmp_path, (episode,))
    manifest = build_analysis_product_manifest(
        snapshot,
        (episode,),
        _prepared_registry(),
        product_id=AnalysisProductId.CLASSIFIER_LADDER,
        target_label_namespace="class",
    )
    output_path = tmp_path / "selections" / "classifier_ladder.json"
    write_analysis_product_manifest(manifest, output_path)
    loaded = AnalysisProductManifest.model_validate_json(output_path.read_text(encoding="utf-8"))
    selection = loaded.episodes[0]

    assert selection.state_view_ids == ()
    assert selection.state_assets == ()
    assert selection.classifier_asset is not None
    assert selection.classifier_asset.path == "classifier/land-episode-1.npz"
    assert selection.target_label_available is True
    assert loaded.policy.requires_prepared_sources is True


def test_classifier_ladder_requires_prepared_source(tmp_path) -> None:
    episode = _episode()
    snapshot = _snapshot(tmp_path, (episode,))
    sources = list(_prepared_registry().sources)
    sources[0] = sources[0].model_copy(
        update={"evidence_state": SourceEvidenceState.FIXTURE_VALIDATED}
    )

    with pytest.raises(ValueError, match="requires prepared source"):
        build_analysis_product_manifest(
            snapshot,
            (episode,),
            _prepared_registry().model_copy(update={"sources": tuple(sources)}),
            product_id=AnalysisProductId.CLASSIFIER_LADDER,
            target_label_namespace="class",
        )


def test_classifier_ladder_requires_explicit_target_namespace(tmp_path) -> None:
    episode = _episode()
    snapshot = _snapshot(tmp_path, (episode,))

    with pytest.raises(ValueError, match="declare a target label namespace"):
        build_analysis_product_manifest(
            snapshot,
            (episode,),
            _prepared_registry(),
            product_id=AnalysisProductId.CLASSIFIER_LADDER,
        )


def test_classifier_ladder_does_not_relabel_route_as_platform_class(tmp_path) -> None:
    base_episode = _episode()
    episode = base_episode.model_copy(
        update={
            "labels": (
                base_episode.labels[0].model_copy(update={"namespace": "route"}),
            )
        }
    )
    snapshot = _snapshot(tmp_path, (episode,))

    with pytest.raises(ValueError, match="target label namespace 'platform_class'"):
        build_analysis_product_manifest(
            snapshot,
            (episode,),
            _prepared_registry(),
            product_id=AnalysisProductId.CLASSIFIER_LADDER,
            target_label_namespace="platform_class",
        )


def test_classifier_ladder_rejects_missing_classifier_view(tmp_path) -> None:
    episode = _episode(classifier=False)
    snapshot = _snapshot(tmp_path, (episode,))

    with pytest.raises(ValueError, match="requires a classifier view"):
        build_analysis_product_manifest(
            snapshot,
            (episode,),
            _prepared_registry(),
            product_id=AnalysisProductId.CLASSIFIER_LADDER,
            target_label_namespace="class",
        )


def test_analysis_product_rejects_registry_snapshot_mismatch(tmp_path) -> None:
    episode = _episode()
    snapshot = _snapshot(tmp_path, (episode,))

    with pytest.raises(ValueError, match="does not match snapshot"):
        build_analysis_product_manifest(
            snapshot,
            (episode,),
            _prepared_registry().model_copy(update={"registry_id": "other-registry"}),
            product_id=AnalysisProductId.KINEMATIC_ANALYSIS,
        )
