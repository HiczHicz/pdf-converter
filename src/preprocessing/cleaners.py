import re
import unicodedata


class UnicodeNormalizer:
    _CHAR_MAP = str.maketrans({
        "–": "-",
        "—": "-",
        "„": '"',
        "”": '"',
        "’": "'",
        "\u00a0": " ",  # twarda spacja z PDF/OCR
    })

    def process(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return text.translate(self._CHAR_MAP)


class Dehyphenator:
    _PATTERN = re.compile(r"([a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ])-\s*\n\s*([a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ])")

    def process(self, text: str) -> str:
        return self._PATTERN.sub(r"\1\2", text)


class BoilerplateRemover:
    def __init__(self, patterns: list[re.Pattern[str]]) -> None:
        self._patterns = patterns

    def process(self, text: str) -> str:
        for pattern in self._patterns:
            text = pattern.sub("", text)
        return text


class WhitespaceCleaner:
    def process(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

class OcrArtifactFixer:
    _SP_ZOO = re.compile(r"\bsp\.\s*z\s*[o0]\.\s*[o0]\.", re.IGNORECASE)

    def process(self, text: str) -> str:
        return self._SP_ZOO.sub("sp. z o.o.", text)