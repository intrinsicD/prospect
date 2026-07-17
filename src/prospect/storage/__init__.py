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
    "InMemoryExperienceStore",
    "LoadedCheckpoint",
    "LedgerIntegrityError",
    "RecordNotFoundError",
    "StorageError",
    "TensorDictExperienceReplay",
    "TorchRLUnavailableError",
    "torchrl_available",
)
