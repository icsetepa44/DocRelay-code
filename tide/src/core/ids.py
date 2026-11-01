"""
Deterministic identifier generation (spec §3). IDs are assigned by application code,
never trusted to the LLM.

  block_id   = {document_id}:p{page:04d}:b{seq:04d}     seq = reading-order index on the page
  stitch_id  = {document_id}:s{seq:06d}                 allocated only when continuation confirmed
  failure_id = {document_id}:failure:{seq:06d}          (deferred Phase B; defined for completeness)
  unit_id    = stitch_id if present else block_id       (assembly, spec §4.15)

The StitchIdAllocator holds a per-document monotonic counter so stitch ids are stable
and unique within a document run.
"""

from __future__ import annotations

from typing import Optional


def block_id(document_id: str, page_number: int, sequence: int) -> str:
    return f"{document_id}:p{page_number:04d}:b{sequence:04d}"


def stitch_id(document_id: str, sequence: int) -> str:
    return f"{document_id}:s{sequence:06d}"


def failure_id(document_id: str, sequence: int) -> str:
    return f"{document_id}:failure:{sequence:06d}"


def unit_id_for(block_id_value: str, stitch_id_value: Optional[str]) -> str:
    return stitch_id_value if stitch_id_value else block_id_value


class StitchIdAllocator:
    """Per-document monotonic allocator for stitch ids."""

    def __init__(self, document_id: str, start: int = 0) -> None:
        self.document_id = document_id
        self._counter = start

    def next(self) -> str:
        self._counter += 1
        return stitch_id(self.document_id, self._counter)

    @property
    def issued(self) -> int:
        return self._counter
