"""v3-facing SAVER metrics API backed by the stable v2 metrics port."""

from saver_v3.metrics import legacy_metrics as _metrics

globals().update(
    {
        name: getattr(_metrics, name)
        for name in dir(_metrics)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("_")]
