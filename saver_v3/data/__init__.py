"""Prepared-data contracts and compact-trace replay helpers for v3 training.

NOTE: prepare_sft_manifest, training_data, dataset, config, prepared_metadata
are NOT eagerly imported here to avoid circular imports with saver_v3.core.
Import them directly: from saver_v3.data.prepare_sft_manifest import ...
"""

from saver_v3.data.compact_trace import compact_trace_row_to_runtime_record, replay_compact_trace_messages
from saver_v3.data.compact_trace_loader import iter_compact_trace_rows, load_compact_trace_rows
from saver_v3.data.compact_trace_replay import (
    CompactTraceStepSFTDataset,
    build_initial_messages,
    expand_compact_trace_row_to_sft_steps,
)
from saver_v3.data.prepared_loader import iter_prepared_rows, load_prepared_rows
from saver_v3.data.prepared_schema import (
    LEGACY_PREPARED_FORMATS,
    PREPARED_FORMAT,
    PREPARED_SCHEMA_VERSION,
    PREPARED_SFT_FORMAT,
    PreparedDataError,
    is_compact_trace_row,
    validate_compact_trace_row,
    validate_prepared_row,
)
from saver_v3.data.runtime_items import (
    build_runtime_item_from_compact_trace_row,
    build_runtime_items_from_compact_trace_rows,
)

__all__ = [
    "CompactTraceStepSFTDataset",
    "LEGACY_PREPARED_FORMATS",
    "PREPARED_FORMAT",
    "PREPARED_SCHEMA_VERSION",
    "PREPARED_SFT_FORMAT",
    "PreparedDataError",
    "build_initial_messages",
    "compact_trace_row_to_runtime_record",
    "expand_compact_trace_row_to_sft_steps",
    "is_compact_trace_row",
    "iter_compact_trace_rows",
    "iter_prepared_rows",
    "load_compact_trace_rows",
    "load_prepared_rows",
    "replay_compact_trace_messages",
    "build_runtime_item_from_compact_trace_row",
    "build_runtime_items_from_compact_trace_rows",
    "validate_compact_trace_row",
    "validate_prepared_row",
]
