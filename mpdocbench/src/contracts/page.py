"""PageData (source for one page) and PageExtraction (structured result for one page)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .block import Block
from .common import to_dict
from .continuation import ContinuationDecision


@dataclass(frozen=True)
class PageData:
    """Source information available for one page (spec §4.1). Produced by the OCR adapter."""

    document_id: str
    page_number: int
    page_image: str          # path to the page image on disk (OCR page_images/ in the MPDocBench fork)
    markdown: str
    html: str
    layout: Dict[str, Any] = field(default_factory=dict)  # Chandra layout: {page_num, page_box, chunks}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PageData":
        return PageData(
            document_id=str(d["document_id"]),
            page_number=int(d["page_number"]),
            page_image=str(d.get("page_image", "")),
            markdown=str(d.get("markdown", "")),
            html=str(d.get("html", "")),
            layout=dict(d.get("layout") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class PageExtraction:
    """Structured extraction of one page (spec §4.8). Block order == reading order."""

    page_number: int
    blocks: List[Block] = field(default_factory=list)
    continuation: Optional[ContinuationDecision] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PageExtraction":
        cont = d.get("continuation")
        return PageExtraction(
            page_number=int(d["page_number"]),
            blocks=[Block.from_dict(b) for b in (d.get("blocks") or [])],
            continuation=ContinuationDecision.from_dict(cont) if cont else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "blocks": [b.to_dict() for b in self.blocks],
            "continuation": self.continuation.to_dict() if self.continuation else None,
        }
