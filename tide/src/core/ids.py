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
    def __init__(self, document_id: str, start: int = 0) -> None:
        self.document_id = document_id
        self._counter = start

    def next(self) -> str:
        self._counter += 1
        return stitch_id(self.document_id, self._counter)

    @property
    def issued(self) -> int:
        return self._counter
