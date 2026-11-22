from pathlib import Path
import fitz          # pip install pymupdf
import pdfplumber    # pip install pdfplumber
from rapidocr_onnxruntime import RapidOCR  # pip install rapidocr-onnxruntime #pip install RapidOCR
import json
import time
import uuid
from pathlib import Path
from preprocessing.patterns import ENTITY_PATTERNS
from preprocessing.base import TextPipeline
from preprocessing.factory import PipelineFactory

class DataExplorer:
    def __init__(self, data_path: str | Path = "./hacknation-2026-main/dane") -> None:
        self.data_path = self._validate_dir(data_path)
        self._ocr_engine = RapidOCR()  # lekki silnik OCR ładowany raz przy starcie

    @staticmethod
    def _validate_dir(path: str | Path) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ścieżka nie istnieje: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Podana ścieżka nie jest katalogiem: {p}")
        return p

    def get_all_dirs(self) -> list[Path]:
        """Zwraca listę ścieżek do wszystkich folderów i podfolderów wewnątrz self.data_path."""
        return [item for item in self.data_path.rglob("*") if item.is_dir()]

    def get_files_in_dir(self, folder_path: str | Path | None = None) -> list[Path]:
        """Zwraca listę plików w podanym folderze (lub w głównym data_path, jeśli nie podano argumentu)."""
        target = self._validate_dir(folder_path) if folder_path is not None else self.data_path
        return [item for item in target.iterdir() if item.is_file()]

    @staticmethod
    def _check_pdf_path(pdf_path: str | Path) -> Path:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Plik PDF nie istnieje: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Podana ścieżka nie jest plikiem: {path}")
        return path

    @staticmethod
    def group_files_by_extension(
        file_paths: list[str | Path],
        allowed_extensions: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        """
        Grupuje ścieżki używając str(p).rsplit(".", 1).
        Dozwolone rozszerzenia (zaczynając od .json) trafiają do swoich kluczy, reszta do 'unknown'.
        """
        if allowed_extensions is None:
            allowed_extensions = [".json", ".pdf", ".csv", ".txt"]

        allowed = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in allowed_extensions
        }

        grouped: dict[str, list[Path]] = {ext: [] for ext in allowed}
        grouped["unknown"] = []

        for p in file_paths:
            path_obj = Path(p)
            parts = str(path_obj).rsplit(".", 1)

            if len(parts) == 2:
                ext = f".{parts[1].lower()}"
            else:
                ext = ""

            if ext in allowed:
                grouped[ext].append(path_obj)
            else:
                grouped["unknown"].append(path_obj)

        return grouped

    @staticmethod
    def filter_by_extension(
        grouped_files: dict[str, list[Path]],
        target_extension: str,
    ) -> list[Path]:
        """Zwraca listę plików dla wskazanego rozszerzenia (lub 'unknown') z wyniku group_files_by_extension."""
        key = target_extension.lower()
        if key != "unknown" and not key.startswith("."):
            key = f".{key}"
        return grouped_files.get(key, [])

    def read_pdf_pymupdf(self, pdf_path: str | Path) -> str:
        """Najszybsza opcja — wyciąga czysty tekst przez PyMuPDF."""
        path = self._check_pdf_path(pdf_path)
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)

    def read_pdf_plumber(self, pdf_path: str | Path) -> str:
        """Najlepsza opcja do dokumentów z tabelami i kolumnami."""
        path = self._check_pdf_path(pdf_path)
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def read_pdf_ocr(self, pdf_path: str | Path, dpi: int = 200) -> str:
        """Odczytuje zeskanowany PDF (obraz) za pomocą lekkiego RapidOCR (bez zewnętrznych instalacji)."""
        path = self._check_pdf_path(pdf_path)
        text_pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                result, _ = self._ocr_engine(pix.tobytes("png"))
                if result:
                    # result to lista elementów: [współrzędne_ramki, tekst, pewność]
                    page_text = "\n".join(line[1] for line in result)
                    text_pages.append(page_text)
        return "\n".join(text_pages)

    def read_pdf(self, pdf_path: str | Path, min_text_length: int = 50) -> str:
        """
        Uniwersalny odczyt PDF:
        1. Próbuje wczytać tekst przez pdfplumber.
        2. Jeśli wyciągnięty tekst (bez spacji) ma mniej niż `min_text_length` znaków
           (czyli PDF jest skanem), automatycznie uruchamia OCR.
        """
        text = self.read_pdf_plumber(pdf_path)
        clean_len = len("".join(text.split()))

        if clean_len < min_text_length:
            return self.read_pdf_ocr(pdf_path)

        return text

    def load_file_json(self, path: str | Path) -> tuple[list, list]:
            p = Path(path)
            with p.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            X = []
            y = []
            if isinstance(raw_data, list):
                for item in raw_data:
                    print(item)
                    X.append(item["text"])
                    y.append(item["label"])
            else:
                X.append(raw_data["text"])
                y.append(raw_data["label"])
            return X, y

    def load_folder_json(self, path: str | Path | None = None) -> tuple[list, list]:
        folder = Path(path) if path is not None else self.data_path
        X = []
        y = []
        for file in sorted(folder.glob("*.json")):
            x, y_ = self.load_file_json(file)
            X.extend(x)
            y.extend(y_)
        return X, y

    def load_file_jsonl(self, path: str | Path) -> tuple[list, list]:
        p = Path(path)
        X = []
        y = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                X.append(item["text"])
                y.append(item["label"])
        return X, y

def _resolve_label(pdf_path: Path, source_dir: Path) -> str | None:
    rel_str = str(pdf_path.relative_to(source_dir)).lower()

    if "negatywn" in rel_str:
        return "negatywne"
    if "przelew" in rel_str or "potwierdzen" in rel_str:
        return "przelew"
    if "odwolan" in rel_str:
        return "odwolanie"
    if "pelnomocnictw" in rel_str:
        return "pelnomocnictwo"

    return None


def extract_entities(text_clean: str) -> dict[str, list[str]]:
    """Wyciąga oryginalne wartości dla wzorców z patterns.py w kolejności maskowania."""
    matched: dict[str, list[str]] = {}
    temp_text = text_clean
    for token, pattern in ENTITY_PATTERNS.items():
        matches = [m.group(0) for m in pattern.finditer(temp_text)]
        if matches:
            matched[token] = matches
            temp_text = pattern.sub(token, temp_text)
    return matched


def build_classification_dataset(
    source_dir: str | Path,
    output_file: str | Path,
    metadata_dir: str | Path,
    explorer: DataExplorer,
    pipeline: TextPipeline | None = None,
    verbose: bool = False,
) -> Path:
    src = Path(source_dir)
    out_file = Path(output_file)
    meta_out = Path(metadata_dir)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    clean_pipeline = PipelineFactory.build_for_extraction()
    nlp_pipeline = pipeline or PipelineFactory.build_for_embeddings()

    records: list[dict[str, str]] = []
    labels_set: set[str] = set()

    pdf_files = sorted(src.rglob("*.pdf"))
    total = len(pdf_files)

    with out_file.open("w", encoding="utf-8") as f:
        for idx, pdf_path in enumerate(pdf_files, start=1):
            rel_path = pdf_path.relative_to(src)
            label = _resolve_label(pdf_path, src)
            if label is None:
                if verbose:
                    print(f"[{idx}/{total}] [POMINIĘTO - brak etykiety]: {rel_path}", flush=True)
                continue

            if verbose:
                print(f"[{idx}/{total}] Przetwarzam ({label}): {rel_path}", flush=True)

            raw_text = explorer.read_pdf(pdf_path)
            text_clean = clean_pipeline.run(raw_text)
            entities = extract_entities(text_clean)
            text_nlp = nlp_pipeline.run(raw_text)

            record_id = uuid.uuid4().hex
            sample_data = {
                "id": record_id,
                "text": text_nlp,          # dla osoby od modelu (TF-IDF)
                "text_clean": text_clean,  # dla Ciebie pod parser JSON
                "entities": entities,      # odzyskane wartości tokenów
                "label": label,
            }
            f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")

            labels_set.add(label)
            records.append({
                "id": record_id,
                "source_pdf": str(pdf_path.resolve()),
                "relative_path": str(rel_path),
                "label": label,
            })

    metadata = {
        "dataset_file": str(out_file.resolve()),
        "labels": sorted(labels_set),
        "total_files": len(records),
        "files": records,
    }
    metadata_path = meta_out / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(f"\nZakończono. Przetworzono {len(records)}/{total} plików.", flush=True)
    return metadata_path

def inspect_pdf_preprocessing(
    pdf_path: str | Path,
    explorer: DataExplorer,
) -> dict[str, str]:
    path = Path(pdf_path)
    print(f"\n{'=' * 60}\nPLIK: {path.name}\n{'=' * 60}")

    t0 = time.perf_counter()
    raw_text = explorer.read_pdf(path)
    t_read = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    text_clean = PipelineFactory.build_for_extraction().run(raw_text)
    t_clean = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    text_masked = PipelineFactory.build_for_embeddings().run(raw_text)
    t_masked = (time.perf_counter() - t2) * 1000

    matched_values = extract_entities(text_clean)

    print(
        f"CZASY [ms]: odczyt={t_read:.1f} | czyszczenie={t_clean:.1f} | "
        f"maskowanie={t_masked:.1f} | RAZEM={t_read + t_masked:.1f} ms"
    )
    print("\nWYKRYTE TOKENY:")
    if not matched_values:
        print("  (brak)")
    for token, values in matched_values.items():
        print(f"  {token} ({len(values)}): {values}")

    print("\n--- OUTPUT Z TOKENAMI ---")
    print(text_masked)

    return {"raw": raw_text, "clean": text_clean, "masked": text_masked}