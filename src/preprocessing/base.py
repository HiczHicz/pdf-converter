from typing import Protocol


class TextStep(Protocol):
    def process(self, text: str) -> str:
        ...


class TextPipeline:
    def __init__(self, steps: list[TextStep]) -> None:
        self._steps = steps

    def run(self, text: str) -> str:
        for step in self._steps:
            text = step.process(text)
        return text