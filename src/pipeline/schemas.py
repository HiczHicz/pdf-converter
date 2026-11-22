from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessingResult:
    source_pdf: Path
    relative_path: Path
    label: str | None = None
    confidence: float | None = None
    output_json: Path | None = None
    data: dict[str, Any] | None = None
    error: str | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "source_pdf": str(self.source_pdf),
            "relative_path": str(self.relative_path),
            "label": self.label,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "output_json": str(self.output_json) if self.output_json else None,
            "error": self.error,
        }