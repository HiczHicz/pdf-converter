import re


class EntityMasker:
    def __init__(self, patterns: dict[str, re.Pattern[str]]) -> None:
        self._patterns = patterns

    def process(self, text: str) -> str:
        for token, pattern in self._patterns.items():
            text = pattern.sub(token, text)
        return text