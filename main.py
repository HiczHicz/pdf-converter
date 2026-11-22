import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import DocumentPipeline  # noqa: E402


def _default_model_path() -> Path:
    candidates = [
        PROJECT_DIR / "src" / "created_models" / "simple_text_classifier.joblib",
        PROJECT_DIR / "models" / "simple_text_classifier.joblib",
        PROJECT_DIR / "simple_text_classifier.joblib",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Klasyfikacja dokumentów PDF i ekstrakcja danych do JSON."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Ścieżka do pojedynczego pliku PDF lub folderu z plikami PDF.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_DIR / "output",
        help="Folder docelowy na wygenerowane pliki JSON (domyślnie: ./output).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=_default_model_path(),
        help="Ścieżka do pliku modelu .joblib (domyślnie: simple_text_classifier.joblib).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimalna pewność klasyfikacji (poniżej traktowane jako 'negatywne').",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Wyłącza logowanie postępu w konsoli.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DocumentPipeline(
        model_path=args.model,
        min_confidence=args.min_confidence,
    )
    pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()