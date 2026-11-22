from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel


class BaseExtractor(ABC):
    """Wspólny kontrakt dla wszystkich ekstraktorów dokumentów."""

    @abstractmethod
    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> BaseModel:
        """Zwraca wypełniony obiekt schematu Pydantic."""

    def extract_dict(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Zwraca słownik gotowy do zapisu w JSON (z zachowaniem aliasów i null)."""
        return self.extract(text_clean, entities).model_dump(by_alias=True)


class ExtractorRegistry:
    """Rejestr mapujący etykietę klasyfikatora na klasę ekstraktora."""

    _registry: dict[str, type[BaseExtractor]] = {}

    @classmethod
    def register(cls, *labels: str) -> Callable[[type[BaseExtractor]], type[BaseExtractor]]:
        def decorator(extractor_cls: type[BaseExtractor]) -> type[BaseExtractor]:
            for label in labels:
                cls._registry[label.lower()] = extractor_cls
            return extractor_cls

        return decorator

    @classmethod
    def get(cls, label: str) -> BaseExtractor | None:
        extractor_cls = cls._registry.get(label.lower())
        return extractor_cls() if extractor_cls else None

    @classmethod
    def supported_labels(cls) -> list[str]:
        return sorted(cls._registry.keys())