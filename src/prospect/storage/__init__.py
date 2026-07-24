"""Storage and persistence substrates for the structured Prospect runtime.

The canonical stores retain linked domain records.  Checkpoint components remain
opaque bytes so persistence does not impose pickle, JSON, tensor, or model-family
requirements on the backend-neutral domain layer.
"""

from __future__ import annotations

from .checkpoint import (
    CheckpointComponent,
    CheckpointComponentManifest,
    CheckpointCoordinator,
    CheckpointFormatError,
    CheckpointIntegrityError,
    CheckpointManifest,
    LoadedCheckpoint,
    checkpoint_manifest_bytes,
    checkpoint_manifest_sha256,
)
from .domain_graph import (
    GRAPH_SCHEMA,
    belief_external_reference,
    decode_domain_graph,
    encode_domain_graph,
    transition_external_reference,
    update_external_reference,
)
from .ledger import EpistemicLedger, LedgerIntegrityError
from .memory import (
    CausalOrderError,
    DuplicateRecordError,
    InMemoryExperienceStore,
    RecordNotFoundError,
    StorageError,
)
from .torchrl_replay import (
    ExperienceTensorCodec,
    TensorDictExperienceReplay,
    TorchRLUnavailableError,
    torchrl_available,
)

__all__ = (
    "CausalOrderError",
    "CheckpointComponent",
    "CheckpointComponentManifest",
    "CheckpointCoordinator",
    "CheckpointFormatError",
    "CheckpointIntegrityError",
    "CheckpointManifest",
    "DuplicateRecordError",
    "EpistemicLedger",
    "ExperienceTensorCodec",
    "GRAPH_SCHEMA",
    "InMemoryExperienceStore",
    "LoadedCheckpoint",
    "LedgerIntegrityError",
    "RecordNotFoundError",
    "StorageError",
    "TensorDictExperienceReplay",
    "TorchRLUnavailableError",
    "belief_external_reference",
    "checkpoint_manifest_bytes",
    "checkpoint_manifest_sha256",
    "decode_domain_graph",
    "encode_domain_graph",
    "torchrl_available",
    "transition_external_reference",
    "update_external_reference",
)
