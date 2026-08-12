from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from preprocessing.schema import LogDataset


@dataclass
class OrthrusAlertCase:
    """Container for ORTHRUS-native inputs used by the *real* ORTHRUS-ano adapter.

    This object exists to decouple the XAI pipeline from ORTHRUS internals while
    still supporting the native ORTHRUS representation:

    - Temporal graph windows are typically represented as `torch_geometric.data.TemporalData`.
    - ORTHRUS encoders often require a `full_data` object containing global edge features.

    IMPORTANT:
    - `temporal_data` and `full_data` are typed as `Any` on purpose.
      This avoids introducing hard runtime dependencies on `torch`/`torch_geometric`
      while we are still scaffolding the integration.
    - The dummy/debug pipeline continues to use `LogDataset` directly.

    Field semantics (expected, but not enforced yet):
    - edge_to_record: maps temporal-data edge index -> original log record index
    - node_to_records: maps node id -> list of original log record indices
    - component_to_edges: maps interpretable component id -> list of edge indices

    Mapping-aware fields (new):
    - mapping_mode: one of {'artifact-only', 'graph-assisted', 'db-assisted'}
    - edge_to_event_uuid: edge index -> ORTHRUS event_uuid (when available)
    - edge_to_log_record: edge index -> original log record payload (when available)
    - edge_to_original_metadata: edge index -> best-effort original metadata (always safe to populate)
    - mapping_quality: free-form metrics about mapping coverage/assumptions
    """

    temporal_data: Any
    full_data: Any | None = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    edge_to_record: Dict[int, int] = field(default_factory=dict)
    node_to_records: Dict[int, List[int]] = field(default_factory=dict)
    component_to_edges: Dict[str, List[int]] = field(default_factory=dict)

    # Mapping awareness
    mapping_mode: str = "artifact-only"
    edge_to_event_uuid: Dict[int, str] = field(default_factory=dict)
    edge_to_log_record: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    edge_to_original_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    mapping_quality: Dict[str, Any] = field(default_factory=dict)

    # Optional link to the debug representation (when available).
    log_dataset: Optional[LogDataset] = None

    @property
    def num_edges(self) -> int:
      td = self.temporal_data
      if td is None:
        return 0
      if hasattr(td, "src"):
        try:
          return int(len(getattr(td, "src")))
        except Exception:
          pass
      if hasattr(td, "edge_index"):
        ei = getattr(td, "edge_index")
        try:
          if hasattr(ei, "shape") and len(getattr(ei, "shape")) == 2:
            return int(ei.shape[1])
          return int(len(ei[0]))
        except Exception:
          pass
      return 0
