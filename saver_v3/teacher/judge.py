"""v3-facing teacher-judge API.

The implementation is a provenance-tracked port of stable v2 teacher-judge
logic. Runtime Qwen execution remains optional; parser, normalization,
packaging, and reweighting helpers are importable without the old v2 tree.
"""

from saver_v3.teacher import teacher_judge as _teacher_judge

globals().update(
    {
        name: getattr(_teacher_judge, name)
        for name in dir(_teacher_judge)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("_")]
