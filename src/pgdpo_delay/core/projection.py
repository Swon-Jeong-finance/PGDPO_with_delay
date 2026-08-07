"""Public projection types for Stage II.

The executable ordering lives in :mod:`pgdpo_delay.core.stage2`; this module
is a compatibility import surface, not a second implementation.
"""

from .stage2 import (
    BlockProjectionStats,
    ProjectionBlocks,
    ProjectionDiagnostics,
    identity_projection,
)

__all__ = [
    "BlockProjectionStats",
    "ProjectionBlocks",
    "ProjectionDiagnostics",
    "identity_projection",
]
