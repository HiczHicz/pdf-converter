
from pathlib import Path
import fitz          # pip install pymupdf
import pdfplumber    # pip install pdfplumber
from rapidocr_onnxruntime import RapidOCR  # pip install rapidocr-onnxruntime #pip install RapidOCR


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