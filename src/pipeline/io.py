import json
from pathlib import Path
from typing import Any

from pipeline.schemas import ProcessingResult


def discover_pdfs(input_path: Path) -> tuple[Path, list[Path]]:
    """Zwraca (katalog_bazowy, lista_pdf) dla pojedynczego pliku lub folderu."""
    path = input_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ścieżka nie istnieje: {path}")

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Oczekiwano pliku .pdf, otrzymano: {path.name}")
        return path.parent, [path]

    pdfs = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    return path, pdfs


class ResultWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_document(self, relative_path: Path, data: dict[str, Any]) -> Path:
        target = self.output_dir / relative_path.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def write_summary(self, results: list[ProcessingResult]) -> Path:
        summary = {
            "total": len(results),
            "extracted": sum(1 for r in results if r.output_json is not None),
            "skipped_no_schema": sum(1 for r in results if r.error is None and r.output_json is None),
            "errors": sum(1 for r in results if r.error is not None),
            "files": [r.to_summary_dict() for r in results],
        }
        target = self.output_dir / "summary.json"
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target