"""Reference SDK for the Open Computation Lifecycle Protocol."""

from oclp.canonical import canonical_json_bytes, record_digest
from oclp.definitions import (
    DefinitionTemplate,
    definition,
    definition_record,
    definition_template,
)
from oclp.models import (
    Artifact,
    ArtifactSet,
    ArtifactSetMember,
    ArtifactSource,
    ComputationDefinition,
    Diagnostic,
    Evidence,
    ExecutionContext,
    GitCheckout,
    GitSource,
    Invocation,
    LifecycleEvent,
    OpaqueSource,
    ServiceSource,
)
from oclp.validation import (
    DerivationValidationError,
    OrchestrationValidationError,
    parse_record,
    validate_derivation_graph,
    validate_invocation_hierarchy,
)

__all__ = [
    "Artifact",
    "ArtifactSource",
    "ArtifactSet",
    "ArtifactSetMember",
    "ComputationDefinition",
    "DefinitionTemplate",
    "Diagnostic",
    "DerivationValidationError",
    "Evidence",
    "ExecutionContext",
    "GitCheckout",
    "GitSource",
    "Invocation",
    "LifecycleEvent",
    "OrchestrationValidationError",
    "OpaqueSource",
    "ServiceSource",
    "canonical_json_bytes",
    "definition",
    "definition_record",
    "definition_template",
    "parse_record",
    "record_digest",
    "validate_derivation_graph",
    "validate_invocation_hierarchy",
]

__version__ = "0.1.0a0"
