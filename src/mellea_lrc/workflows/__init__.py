"""The three user-defined, highest-level Document workflows.

Individual extraction and validation stages stay independently callable.
Adding another workflow requires the user's explicit design decision, together
with one corresponding workflow report in ``evaluations``.
"""

from mellea_lrc.workflows.grow_leaves import grow_leaves
from mellea_lrc.workflows.grow_roots import grow_roots
from mellea_lrc.workflows.validate_roots import validate_roots

__all__ = ["grow_leaves", "grow_roots", "validate_roots"]
