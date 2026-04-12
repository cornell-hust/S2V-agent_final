"""Runtime entrypoints for the idea2_v3 SAVER package.

Only a constrained subset of the package is owned in this workspace. The
modules exported here intentionally stay thin and dependency-light so the
package can be imported even when heavyweight training or inference
dependencies are missing from the current Python environment.
"""

__all__ = [
    "cli",
    "inference",
    "rl",
    "sft",
]

