from pathlib import Path


class DataExplorer:
    def __init__(self, data_path: str | Path = "./hacknation-2026-main/dane") -> None:
        self.data_path = self._validate_dir(data_path)

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