import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extraction import ExtractorRegistry


def flatten_json(data: Any, prefix: str = "") -> dict[str, Any]:
    """Spłaszcza zagnieżdżony JSON do mapy 'klucz.podklucz[0].pole' -> wartość."""
    items: dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(flatten_json(v, new_key))
    elif isinstance(data, list):
        if not data:
            items[prefix] = []
        for idx, v in enumerate(data):
            new_key = f"{prefix}[{idx}]"
            items.update(flatten_json(v, new_key))
    else:
        items[prefix] = data
    return items


def _normalize_value(val: Any) -> Any:
    if isinstance(val, str):
        return " ".join(val.split()).strip().lower()
    return val


@dataclass
class FieldDiff:
    file_name: str
    label: str
    field_path: str
    expected: Any
    predicted: Any


@dataclass
class EvaluationReport:
    total_files: int = 0
    total_fields: int = 0
    matched_fields: int = 0
    per_label: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_field_hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_field_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    diffs: list[FieldDiff] = field(default_factory=list)
    missing_gt: list[str] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return (self.matched_fields / self.total_fields) if self.total_fields else 0.0

    def print_summary(self, max_diffs: int = 20) -> None:
        print(f"=== EWALUACJA ({self.total_files} plików) ===")
        print(f"Overall field accuracy: {self.overall_accuracy:.2%} ({self.matched_fields}/{self.total_fields})")

        if self.missing_gt:
            print(f"\n[UWAGA] Brak wzorcowego .json dla {len(self.missing_gt)} plików (pierwsze 5):")
            for path in self.missing_gt[:5]:
                print(f"  - {path}")

        print("\n--- Per klasa dokumentu ---")
        for label, scores in sorted(self.per_label.items()):
            avg = sum(scores) / len(scores)
            print(f"  {label:<18}: {avg:.2%} ({len(scores)} plików)")

        print("\n--- Najsłabsze pola ---")
        field_scores = [
            (self.per_field_hits[f] / self.per_field_total[f], f, self.per_field_hits[f], self.per_field_total[f])
            for f in self.per_field_total
        ]
        for acc, fname, hits, tot in sorted(field_scores)[:15]:
            print(f"  {fname:<45}: {acc:.2%} ({hits}/{tot})")

        if self.diffs:
            print(f"\n--- Przykładowe rozbieżności (max {max_diffs}) ---")
            for d in self.diffs[:max_diffs]:
                print(f"  [{d.label}] {d.file_name} | {d.field_path}")
                print(f"      expected : {d.expected!r}")
                print(f"      predicted: {d.predicted!r}")


class ExtractionEvaluator:
    def compare_dicts(
        self,
        predicted: dict[str, Any],
        expected: dict[str, Any],
        file_name: str,
        label: str,
        report: EvaluationReport,
    ) -> float:
        pred_flat = flatten_json(predicted)
        exp_flat = flatten_json(expected)
        all_keys = sorted(set(pred_flat.keys()) | set(exp_flat.keys()))

        hits = 0
        for key in all_keys:
            exp_val = exp_flat.get(key)
            pred_val = pred_flat.get(key)
            is_match = _normalize_value(exp_val) == _normalize_value(pred_val)

            report.total_fields += 1
            report.per_field_total[f"{label}:{key}"] += 1
            if is_match:
                hits += 1
                report.matched_fields += 1
                report.per_field_hits[f"{label}:{key}"] += 1
            else:
                report.diffs.append(FieldDiff(file_name, label, key, exp_val, pred_val))

        file_acc = hits / len(all_keys) if all_keys else 1.0
        report.total_files += 1
        report.per_label[label].append(file_acc)
        return file_acc

    # --- TYMCZASOWA METODA POD ZBIÓR TRENINGOWY (do usunięcia po testach) ---
    @staticmethod
    def _resolve_training_gt_path(
        meta_item: dict[str, str],
        ground_truth_root: Path | None,
    ) -> Path:
        """
        Zwraca ścieżkę do wzorcowego .json odpowiadającego PDF-owi w tym samym podfolderze.
        np. .../przyklad_01/przelew.pdf -> .../przyklad_01/przelew.json
        """
        if ground_truth_root is not None:
            return ground_truth_root / Path(meta_item["relative_path"]).with_suffix(".json")
        return Path(meta_item["source_pdf"]).with_suffix(".json")

    def evaluate_training_from_metadata(
        self,
        dataset_jsonl: str | Path,
        metadata_json: str | Path,
        ground_truth_root: str | Path | None = None,
        save_predictions_dir: str | Path | None = None,
        only_labels: set[str] | None = None,
    ) -> EvaluationReport:
        """
        Iteruje po dataset.jsonl, pobiera ścieżkę źródłowego PDF-a z metadata.json,
        szuka wzorca .json w tym samym podfolderze i porównuje z wynikiem ekstraktora.
        """
        meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
        id_to_meta = {item["id"]: item for item in meta["files"]}

        gt_root = Path(ground_truth_root) if ground_truth_root else None
        out_dir = Path(save_predictions_dir) if save_predictions_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        report = EvaluationReport()

        with Path(dataset_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                label = record["label"]

                if only_labels and label not in only_labels:
                    continue

                extractor = ExtractorRegistry.get(label)
                if extractor is None:
                    continue  # pomija np. 'negatywne'

                meta_item = id_to_meta.get(record["id"])
                if not meta_item:
                    report.missing_gt.append(f"brak w metadata.json: id={record['id']}")
                    continue

                rel_path = Path(meta_item["relative_path"])
                gt_path = self._resolve_training_gt_path(meta_item, gt_root)

                predicted = extractor.extract_dict(
                    text_clean=record["text_clean"],
                    entities=record.get("entities"),
                )

                if out_dir:
                    pred_file = out_dir / rel_path.with_suffix(".json")
                    pred_file.parent.mkdir(parents=True, exist_ok=True)
                    pred_file.write_text(
                        json.dumps(predicted, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                if not gt_path.is_file():
                    report.missing_gt.append(str(gt_path))
                    continue

                expected = json.loads(gt_path.read_text(encoding="utf-8"))
                self.compare_dicts(predicted, expected, str(rel_path), label, report)

        return report