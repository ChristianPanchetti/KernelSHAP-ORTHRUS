from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pytest

from adapters.orthrus_ano_adapter import DummyOrthrusAnoAdapter
from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder
from preprocessing.log_input_loader import LogInputLoader
from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_alert_case_loader import OrthrusAlertCaseLoader
from preprocessing.orthrus_mapping import (
    ArtifactOnlyMappingProvider,
    DbAssistedMappingProvider,
    GraphAssistedMappingProvider,
    describe_component,
)


class FakeTemporalData:
    """Torch/PyG-free TemporalData-like object for unit tests."""

    def __init__(self, src, dst, t, edge_type=None):
        self.src = list(src)
        self.dst = list(dst)
        self.t = list(t)
        if edge_type is not None:
            self.edge_type = list(edge_type)


def test_artifact_only_mapping_provider_enriches_case_and_describes_edges():
    td = FakeTemporalData(
        src=[1, 1, 2],
        dst=[2, 3, 3],
        t=[100, 120, 180],
        edge_type=[0, 1, 1],
    )

    case = OrthrusAlertCase(temporal_data=td)
    provider = ArtifactOnlyMappingProvider()
    provider.enrich_case(case)

    assert case.mapping_mode == "artifact-only"
    assert case.mapping_quality.get("mode") == "artifact-only"

    assert len(case.edge_to_original_metadata) == 3
    assert case.edge_to_original_metadata[0]["src"] == 1
    assert case.edge_to_original_metadata[0]["dst"] == 2
    assert case.edge_to_original_metadata[0]["t"] == 100
    assert case.edge_to_original_metadata[0]["edge_type"] == 0

    desc0 = provider.describe_edge(case, 0)
    assert "edge=0" in desc0
    assert "src=1" in desc0
    assert "dst=2" in desc0
    assert "edge_type=0" in desc0


def test_db_requires_a_data_source_and_graph_provider_remains_stubbed():
    td = FakeTemporalData(src=[1], dst=[2], t=[100], edge_type=[0])
    case = OrthrusAlertCase(temporal_data=td)

    with pytest.raises(NotImplementedError):
        DbAssistedMappingProvider().enrich_case(case)

    with pytest.raises(NotImplementedError):
        GraphAssistedMappingProvider().enrich_case(case)


def test_db_assisted_fields_present_but_empty_by_default():
    td = FakeTemporalData(src=[1], dst=[2], t=[100], edge_type=[0])
    case = OrthrusAlertCase(temporal_data=td, mapping_mode="db-assisted")

    assert case.mapping_mode == "db-assisted"
    assert case.edge_to_event_uuid == {}
    assert case.edge_to_log_record == {}
    assert case.edge_to_original_metadata == {}
    assert case.mapping_quality == {}

    # Description should not crash even if db-assisted mapping isn't available yet.
    case.component_to_edges = {"edge:0": [0]}
    text = describe_component(case, "edge:0")
    assert "db-assisted" in text


def test_builder_is_independent_of_mapping_fields():
    td = FakeTemporalData(
        src=[1, 1, 2],
        dst=[2, 3, 3],
        t=[100, 120, 180],
        edge_type=[0, 1, 1],
    )

    case_a = OrthrusAlertCase(temporal_data=td, mapping_mode="artifact-only")
    case_b = OrthrusAlertCase(
        temporal_data=td,
        mapping_mode="db-assisted",
        edge_to_event_uuid={0: "uuid-0"},
        edge_to_log_record={0: {"dummy": True}},
        mapping_quality={"dummy": True},
    )

    builder = OrthrusInterpretableBuilder(grouping_mode="edge_type")
    builder.build(case_a)
    builder.build(case_b)

    assert case_a.component_to_edges == case_b.component_to_edges
    assert set(case_a.component_to_edges.keys()) == {"edge_type:0", "edge_type:1"}


def test_describe_component_artifact_only_includes_edge_samples():
    td = FakeTemporalData(
        src=[1, 1, 2],
        dst=[2, 3, 3],
        t=[100, 120, 180],
        edge_type=[0, 1, 1],
    )

    case = OrthrusAlertCase(temporal_data=td)
    ArtifactOnlyMappingProvider().enrich_case(case)

    OrthrusInterpretableBuilder(grouping_mode="edge_type").build(case)

    text = describe_component(case, "edge_type:1", max_edges=2)
    assert "component=edge_type:1" in text
    assert "edges=2" in text
    assert "edge=1" in text


def test_alert_case_loader_artifact_only_works_with_pickle(tmp_path: Path):
    td = FakeTemporalData(
        src=[1, 1, 2],
        dst=[2, 3, 3],
        t=[100, 120, 180],
        edge_type=[0, 1, 1],
    )

    p = tmp_path / "temporal_data.synthetic"
    with p.open("wb") as f:
        pickle.dump(td, f)

    loader = OrthrusAlertCaseLoader(logger=logging.getLogger("test"))
    case = loader.load(p, mapping_mode="artifact-only")

    assert case.mapping_mode == "artifact-only"
    assert len(case.edge_to_original_metadata) == 3
    assert case.edge_to_original_metadata[2]["src"] == 2


def test_dummy_mode_still_scores_log_dataset(default_input_path: Path):
    dataset = LogInputLoader(logger=logging.getLogger("test")).load(default_input_path)
    score = DummyOrthrusAnoAdapter(seed=0, logger=logging.getLogger("test")).predict_anomaly_score(dataset)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
