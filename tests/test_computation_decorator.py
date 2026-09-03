"""Tests for callable-bound OCLP Computation declarations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from oclp import (
    ArtifactHandle,
    ArtifactSetHandle,
    ArtifactSetInput,
    CsvArtifact,
    GitSource,
    JsonArtifact,
    OclpRun,
    active_run,
    artifact_set_input,
    computation,
    computation_input_artifact_types,
    computation_record,
    computation_template,
    evaluate_evidence,
    evidence,
    evidence_implementation,
    json_artifact,
    lifecycle,
    lifecycle_template,
    load_release_manifest,
    many,
    observe_lifecycle,
    record_digest,
    validate_derivation_graph,
)
from oclp.models import (
    ArtifactSet,
    Event,
    Evidence,
    Execution,
    ParameterDefinition,
    PortDefinition,
    RecordReference,
)
from oclp.publishing import LocalArtifactPublisher


@computation(
    id="urn:example:computation:normalize-report",
    name="Normalize report",
    input_ports=(PortDefinition(name="source", media_types=("application/json",)),),
    output_ports=(PortDefinition(name="report", media_types=("application/json",)),),
)
def normalize_report(source: str) -> str:
    """An ordinary callable with colocated OCLP Computation metadata."""

    return source.strip()


@computation(
    id="urn:example:computation:use-release",
    name="Use validated release",
    inputs={
        "release": artifact_set_input({"configuration": JsonArtifact}),
    },
)
def use_release(release: ArtifactSetHandle) -> str:
    """Consume a named JSON member from one exact ArtifactSet input."""

    configuration = release.load_member("configuration", dict[str, str])
    assert isinstance(configuration, dict)
    return str(configuration["source"])


@lifecycle(
    namespace="urn:example",
    name="Report normalization lifecycle",
)
def normalize_report_lifecycle(source: str) -> str:
    """A workflow whose real child Computation is observed by the SDK."""

    assert active_run() is not None
    return normalize_report(source)


def test_computation_decorator_keeps_callable_behavior_and_derives_locator() -> None:
    record = computation_record(
        normalize_report,
        source=GitSource(
            repository="https://github.com/example/reports.git",
            commit="a" * 40,
            path="src/reports.py",
        ),
    )

    assert normalize_report(" report ") == "report"
    assert computation_template(normalize_report).id == record.id
    assert record.implementation.locator.endswith(".normalize_report")
    assert record.input_ports[0].name == "source"
    assert record.output_ports[0].name == "report"


def test_oclp_run_exposes_the_computation_for_an_observed_result(tmp_path) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="computation-binding",
            source=source,
        ) as observed:
            result = normalize_report(" report ")
            computation = observed.computation_for(result)

    assert computation.id == "urn:example:computation:normalize-report"
    assert computation.digest == record_digest(
        computation_record(normalize_report, source=source)
    )


def test_observe_lifecycle_derives_one_shared_profile_for_real_executions(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with observe_lifecycle(
            normalize_report_lifecycle,
            publisher=publisher,
            run_id="report-run-1",
            source=source,
        ) as observed:
            result = normalize_report_lifecycle(" report ")
            execution_ref = observed.execution_for(result)
        records = publisher.records()

    assert lifecycle_template(normalize_report_lifecycle).name == (
        "Report normalization lifecycle"
    )
    execution = next(
        record
        for record in records
        if isinstance(record, Execution) and record.id == execution_ref.id
    )
    assert execution.profiles == {
        "lifecycle": {
            "version": "0.2.0-draft",
            "run_id": "urn:example:lifecycle:report-run-1",
            "run_name": "Report normalization lifecycle",
        }
    }
    assert execution.name == "Normalize report"


def test_computation_template_requires_decorated_callable() -> None:
    def undecorated() -> None:
        pass

    with pytest.raises(ValueError, match="has no OCLP Computation template"):
        computation_template(undecorated)


def test_computation_rejects_input_ports_without_matching_parameters() -> None:
    with pytest.raises(ValueError, match="do not match parameters"):

        @computation(
            id="urn:example:computation:input-mismatch",
            name="Input mismatch",
            input_ports=(PortDefinition(name="source_snapshot"),),
        )
        def input_mismatch(source: str) -> str:
            return source


def test_computation_derives_port_metadata_from_an_artifact_type() -> None:
    @computation(
        id="urn:example:computation:read-csv",
        name="Read CSV",
        inputs={"source_snapshot": CsvArtifact},
    )
    def read_csv(source_snapshot: str) -> str:
        return source_snapshot

    template = computation_template(read_csv)

    assert template.input_ports == (
        PortDefinition(name="source_snapshot", media_types=("text/csv",)),
    )
    assert computation_input_artifact_types(read_csv) == {
        "source_snapshot": CsvArtifact,
    }


def test_computation_binds_and_validates_one_artifact_set_input(tmp_path) -> None:
    assert isinstance(
        computation_input_artifact_types(use_release)["release"], ArtifactSetInput
    )
    assert computation_template(use_release).input_ports == (
        PortDefinition(name="release"),
    )

    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="release-consumer",
            source=source,
        ) as observed:
            configuration = source_configuration()
            release = observed.publish_artifact_set(
                key="validated-release",
                name="Validated release",
                members={"configuration": (configuration, "config")},
            )
            result = use_release(release)
            execution = observed.execution_for(result)
        records = publisher.records()

    assert result == "hourly-bike-data"
    assert execution.id == "urn:example:execution:use-release:release-consumer"
    execution_record = next(
        record
        for record in records
        if isinstance(record, Execution) and record.id == execution.id
    )
    assert execution_record.inputs == {"release": (release.reference,)}


@json_artifact(name="Source configuration")
def source_configuration() -> dict[str, str]:
    return {"source": "hourly-bike-data"}


@json_artifact(name="Validation report")
def validation_report() -> dict[str, float]:
    return {"rmse": 0.2}


def test_oclp_run_publishes_an_artifact_set_from_exact_handles(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="release-1",
            source=source,
            profiles={
                "lifecycle": {
                    "version": "0.2.0-draft",
                    "run_id": "urn:example:lifecycle:release-1",
                    "run_name": "Validated release lifecycle",
                }
            },
        ) as observed:
            configuration = source_configuration()
            evaluation = validation_report()
            release = observed.publish_artifact_set(
                key="validated-release",
                name="Validated release",
                members={
                    "configuration": (configuration, "config"),
                    "evaluation": (evaluation, "validation"),
                },
            )
        records = publisher.records()

    assert isinstance(release, ArtifactSetHandle)
    assert release.artifact_set.id == (
        "urn:example:artifact-set:validated-release:release-1"
    )
    assert [member.name for member in release.artifact_set.members] == [
        "configuration",
        "evaluation",
    ]
    assert [member.role for member in release.artifact_set.members] == [
        "config",
        "validation",
    ]
    assert release.artifact_set.profiles == {
        "lifecycle": {
            "version": "0.2.0-draft",
            "run_id": "urn:example:lifecycle:release-1",
            "run_name": "Validated release lifecycle",
        }
    }
    assert not any(isinstance(record, Execution) for record in records)
    assert any(isinstance(record, ArtifactSet) for record in records)
    validate_derivation_graph(records)


def test_oclp_run_materializes_a_release_manifest_from_exact_handles(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="b" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="release-manifest-1",
            source=source,
        ) as observed:
            configuration = source_configuration()
            evaluation = validation_report()
            release = observed.publish_artifact_set(
                key="validated-release",
                name="Validated release",
                members={
                    "configuration": (configuration, "config"),
                    "evaluation": (evaluation, "validation"),
                },
                materialize_manifest=True,
                manifest_name="Validated release manifest",
            )
        records = publisher.records()

    assert release.manifest is not None
    assert release.manifest.artifact.name == "Validated release manifest"
    assert release.manifest.artifact.media_type == "application/json"
    assert release.manifest.path.name == "release-manifest.json"
    assert [member.name for member in release.artifact_set.members] == [
        "configuration",
        "evaluation",
    ]

    manifest = json.loads(release.manifest.read_verified_bytes())
    assert manifest["artifact_set"]["reference"] == release.reference.model_dump(
        mode="json"
    )
    assert manifest["artifact_set"]["record"] == release.artifact_set.model_dump(
        mode="json"
    )
    manifest_members = manifest["artifact_set"]["record"]["members"]
    assert [member["name"] for member in manifest_members] == [
        "configuration",
        "evaluation",
    ]
    assert {entry["record"]["id"] for entry in manifest["records"]} == {
        configuration.artifact.id,
        evaluation.artifact.id,
    }
    assert not any(isinstance(record, Execution) for record in records)
    assert any(
        record.id == release.manifest.artifact.id for record in records
    )
    validate_derivation_graph(records)

    loaded = load_release_manifest(release.manifest.path)
    assert loaded.reference == release.reference
    assert loaded.artifact_set == release.artifact_set
    assert loaded.member("configuration").reference == configuration.reference
    assert loaded.load_member("configuration", dict[str, str]) == {
        "source": "hourly-bike-data"
    }


@computation(
    id="urn:example:computation:parameterized-report",
    name="Produce a parameterized report",
    input_ports=(PortDefinition(name="source"),),
    outputs={"report": JsonArtifact(name="Parameterized report")},
)
def parameterized_report(
    source: str,
    *,
    fold_number: int,
    mode: Literal["fast", "full"] = "fast",
    scratch_path: Path,
) -> dict[str, object]:
    """Exercise inferred JSON parameters and local-only plumbing."""

    return {
        "source": source,
        "fold": fold_number,
        "mode": mode,
        "scratch": scratch_path.name,
    }


def test_computation_inferrs_json_parameter_contract_from_callable_signature(
    tmp_path,
) -> None:
    template = computation_template(parameterized_report)

    assert template.parameter_definitions == (
        ParameterDefinition(
            name="fold_number",
            schema={"type": "integer"},
        ),
        ParameterDefinition(
            name="mode",
            schema={"type": "string", "enum": ["fast", "full"], "default": "fast"},
            required=False,
        ),
    )

    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="parameter-contract",
            source=source,
        ):
            parameterized_report(
                "source text",
                fold_number=2,
                scratch_path=tmp_path / "scratch",
            )
        execution = next(
            record
            for record in publisher.records()
            if isinstance(record, Execution)
            and record.id
            == "urn:example:execution:parameterized-report:parameter-contract"
        )

    assert execution.parameters == {"fold_number": 2, "mode": "fast"}


@json_artifact(
    id="urn:example:artifact:fold-definition",
    name="Example fold definition",
)
def fetch_fold_definition() -> dict[str, object]:
    """Provide generic JSON rather than a pandas table representation."""

    return {"folds": [{"fold": 1, "train_end": "2024-01-01"}]}


@computation(
    id="urn:example:computation:read-fold-definition",
    name="Read fold definition",
    inputs={"fold_definition": JsonArtifact},
)
def read_fold_definition(fold_definition: dict[str, object]) -> int:
    """Prove that the runtime annotation, not JSON storage, selects mapping."""

    folds = fold_definition["folds"]
    assert isinstance(folds, list)
    return len(folds)


@json_artifact(name="Example report")
def fetch_report(value: int) -> dict[str, int]:
    """Acquire a small JSON value used to test a many-Artifact input."""

    return {"value": value}


@evidence(
    name="Positive report total",
)
def positive_total(summary: dict[str, int]) -> str:
    return "pass" if summary["total"] > 0 else "fail"


@computation(
    id="urn:example:computation:aggregate-reports",
    name="Aggregate reports",
    inputs={"reports": many(JsonArtifact)},
    outputs={"summary": JsonArtifact(name="Report summary")},
    requires=(positive_total,),
)
def aggregate_reports(reports: tuple[dict[str, int], ...]) -> dict[str, dict[str, int]]:
    """Prove that many JSON Artifacts arrive as ordinary typed mappings."""

    return {"summary": {"total": sum(report["value"] for report in reports)}}


def test_json_artifact_adapts_to_a_mapping_when_the_callable_requests_one(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="json-mapping",
            source=source,
        ):
            fold_definition = fetch_fold_definition()
            count = read_fold_definition(fold_definition)

        records = publisher.records()

    assert isinstance(fold_definition, ArtifactHandle)
    assert count == 1
    record = next(
        record
        for record in records
        if record.id == "urn:example:execution:read-fold-definition:json-mapping"
    )
    assert isinstance(record, Execution)
    assert record.inputs == {"fold_definition": (fold_definition.reference,)}


def test_active_run_binds_many_artifacts_and_evaluates_required_evidence(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="many-artifacts",
            source=source,
        ) as observed:
            reports = (fetch_report(2), fetch_report(3))
            result = aggregate_reports(reports)
            execution = observed.execution_for(result)
            outputs = observed.outputs_for(result)
            evidence = observed.evidence_for(result)
        records = publisher.records()

    assert result == {"summary": {"total": 5}}
    assert computation_template(aggregate_reports).input_ports == (
        PortDefinition(
            name="reports",
            cardinality="many",
            media_types=("application/json",),
        ),
    )
    execution_record = next(
        record
        for record in records
        if isinstance(record, Execution) and record.id == execution.id
    )
    assert execution_record.inputs == {
        "reports": tuple(report.reference for report in reports),
    }
    assert execution_record.outputs == {"summary": (outputs["summary"].reference,)}
    assert len(evidence) == 1
    assert evidence[0].subject == execution
    assert any(
        isinstance(record, Evidence)
        and record.id == evidence[0].id
        and record.outcome == "pass"
        for record in records
    )


def test_computation_rejects_mixed_raw_and_artifact_input_declarations() -> None:
    with pytest.raises(ValueError, match="either input_ports or inputs"):
        computation(
            id="urn:example:computation:ambiguous-inputs",
            name="Ambiguous inputs",
            input_ports=(PortDefinition(name="source"),),
            inputs={"source": CsvArtifact},
        )


def test_computation_rejects_duplicate_ports_when_declared() -> None:
    with pytest.raises(ValidationError, match="port names must be unique"):
        computation(
            id="urn:example:computation:invalid",
            name="Invalid computation",
            input_ports=(PortDefinition(name="source"), PortDefinition(name="source")),
        )


def test_computation_requires_a_concrete_persisted_output_representation() -> None:
    with pytest.raises(TypeError, match="concrete Artifact"):
        computation(
            id="urn:example:computation:implicit-output",
            name="Implicit output",
            outputs=("result",),  # type: ignore[arg-type]
        )


def test_computation_requires_application_owned_output_names() -> None:
    with pytest.raises(ValueError, match="application-supplied ArtifactType names"):
        computation(
            id="urn:example:computation:unnamed-output",
            name="Produce unnamed output",
            outputs={"result": JsonArtifact()},
        )


def test_materialized_artifact_set_requires_an_application_owned_manifest_name(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="c" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="unnamed-manifest",
            source=source,
        ) as observed:
            configuration = source_configuration()
            with pytest.raises(ValueError, match="application-supplied manifest_name"):
                observed.publish_artifact_set(
                    key="validated-release",
                    name="Validated release",
                    members={"configuration": (configuration, "config")},
                    materialize_manifest=True,
                )


@evidence(
    name="Quality check",
)
def quality_gate(value: int) -> str:
    return "pass" if value > 0 else "fail"


@computation(
    id="urn:example:computation:quality-checked",
    name="Quality checked",
    requires=(quality_gate,),
)
def quality_checked() -> None: ...


def test_computation_decorator_materializes_required_evidence() -> None:
    template = computation_template(quality_checked)
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    record = computation_record(quality_checked, source=source)

    assert template.required_evaluators == (quality_gate,)
    assert record.required_evidence == (
        evidence_implementation(quality_gate, source=source),
    )


def test_evidence_decorator_evaluates_and_binds_the_source_bound_evaluator() -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    subject = RecordReference(
        id="urn:example:execution:test",
        digest=record_digest(computation_record(normalize_report, source=source)),
    )
    record = evaluate_evidence(
        quality_gate,
        3,
        subject=subject,
        source=source,
        id="urn:example:evidence:quality:test",
        observed_at="2026-08-30T18:00:00Z",
    )

    assert record.outcome == "pass"
    assert record.evaluator == evidence_implementation(quality_gate, source=source)


@evidence(name="Broken summary check")
def broken_summary_check(summary: dict[str, int]) -> str:
    raise RuntimeError(f"cannot evaluate {summary['total']}")


@evidence(name="Passing summary check")
def passing_summary_check(summary: dict[str, int]) -> str:
    return "pass"


@computation(
    id="urn:example:computation:collect-evidence",
    name="Collect all Evidence",
    outputs={"summary": JsonArtifact(name="Evidence summary")},
    requires=(broken_summary_check, passing_summary_check),
)
def collect_evidence() -> dict[str, dict[str, int]]:
    return {"summary": {"total": 3}}


def test_runtime_collects_all_required_evidence_when_a_gate_errors(tmp_path) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="collect-all-evidence",
            source=source,
        ) as observed:
            result = collect_evidence()
            evidence = observed.evidence_for(result)
        terminal = next(
            record
            for record in publisher.records()
            if isinstance(record, Event) and record.event_type == "execution-terminal"
        )

    assert [record.outcome for record in evidence] == ["error", "pass"]
    assert evidence[0].diagnostic is not None
    assert terminal.status == "failed"


class CsvTable:
    """Small pandas-shaped value without making pandas an SDK dependency."""

    def __init__(self, rows: int) -> None:
        self.rows = rows

    def to_csv(self, *, index: bool, lineterminator: str) -> str:
        assert index is False
        return "value" + lineterminator + "1" + lineterminator


@dataclass(frozen=True)
class PreparedTable:
    table: CsvTable
    metadata: dict[str, int]


@computation(
    id="urn:example:computation:prepare-table",
    name="Prepare table",
    outputs={
        "table": CsvArtifact(
            name="Prepared source table",
            key="prepared-table",
            path="prepared/table.csv",
            schema_uri="urn:example:schema:table:v1",
        ),
        "metadata": JsonArtifact(
            name="Prepared table metadata",
            key="prepared-table-metadata",
            path="prepared/metadata.json",
            annotations={"producer": "test"},
        ),
    },
)
def prepare_table() -> PreparedTable:
    return PreparedTable(table=CsvTable(rows=275), metadata={"rows": 275})


@computation(
    id="urn:example:computation:fetch-table",
    name="Fetch table",
    outputs={"source_snapshot": CsvArtifact(name="Source table")},
)
def fetch_table(dataset_id: int = 275) -> CsvTable:
    return CsvTable(rows=dataset_id)


@computation(
    id="urn:example:computation:summarize-table",
    name="Summarize table",
    input_ports=(PortDefinition(name="source", media_types=("text/csv",)),),
    outputs={"summary": JsonArtifact(name="Table summary")},
)
def summarize_table(source: CsvTable) -> dict[str, int]:
    return {"rows": source.rows}


def test_active_run_materializes_return_values_and_tracks_exact_input_objects(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="automatic-output",
            source=source,
        ) as observed:
            table = fetch_table()
            summary = summarize_table(table)
            source_snapshot = observed.artifact_for(table, port="source_snapshot")
            ingest_execution = observed.execution_for(table)

        records = publisher.records()

    assert summary == {"rows": 275}
    assert source_snapshot.path.read_text() == "value\n1\n"
    assert source_snapshot.artifact.id == (
        "urn:example:artifact:fetch-table:automatic-output:source_snapshot"
    )
    assert ingest_execution.id == "urn:example:execution:fetch-table:automatic-output"

    executions = [record for record in records if isinstance(record, Execution)]
    fetch_execution = next(
        record for record in executions if record.id == ingest_execution.id
    )
    summary_execution = next(
        record
        for record in executions
        if record.id == "urn:example:execution:summarize-table:automatic-output"
    )
    assert fetch_execution.parameters == {"dataset_id": 275}
    assert fetch_execution.outputs == {"source_snapshot": (source_snapshot.reference,)}
    assert computation_template(fetch_table).output_ports[0].media_types == (
        "text/csv",
    )
    assert summary_execution.inputs == {"source": (source_snapshot.reference,)}
    assert summary_execution.outputs is not None
    assert any(
        isinstance(record, Event)
        and record.execution == ingest_execution
        and record.event_type == "execution-terminal"
        and record.status == "succeeded"
        for record in records
    )


def test_computation_output_declarations_own_metadata_and_output_bindings(
    tmp_path,
) -> None:
    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="prepared-output",
            source=source,
        ) as observed:
            prepared = prepare_table()
            outputs = observed.outputs_for(prepared)
            execution = observed.execution_for(prepared)

        records = publisher.records()

    table = outputs["table"].artifact
    metadata = outputs["metadata"]
    assert isinstance(outputs["table"], ArtifactHandle)
    assert isinstance(metadata, ArtifactHandle)
    assert table.id == "urn:example:artifact:prepared-table:prepared-output"
    assert table.name == "Prepared source table"
    assert table.schema_uri == "urn:example:schema:table:v1"
    assert table.locations[0].endswith("/prepared/table.csv")
    assert metadata.artifact.profiles is None
    assert metadata.artifact.annotations == {"producer": "test"}
    assert metadata.path.read_text() == '{\n  "rows": 275\n}\n'
    execution_record = next(
        record
        for record in records
        if isinstance(record, Execution) and record.id == execution.id
    )
    assert execution_record.outputs == {
        "table": (outputs["table"].reference,),
        "metadata": (metadata.reference,),
    }
