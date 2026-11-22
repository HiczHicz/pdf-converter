from pathlib import Path
import joblib
import numpy as np


class DocumentClassifier:
    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Nie znaleziono modelu: {path.resolve()}")

        loaded = joblib.load(path)
        self._boost_intro_words = 0

        if isinstance(loaded, dict) and "pipeline" in loaded:
            self._pipeline = loaded["pipeline"]
            self._boost_intro_words = int(loaded.get("boost_intro_words", 0))
        else:
            self._pipeline = loaded

    def _prepare(self, text: str) -> str:
        if self._boost_intro_words > 0:
            words = text.split()
            intro = " ".join(words[: self._boost_intro_words])
            return f"{intro} {text}"
        return text

    def predict(self, text_nlp: str) -> tuple[str, float | None]:
        prepared = [self._prepare(text_nlp)]
        if hasattr(self._pipeline, "predict_proba"):
            probs = self._pipeline.predict_proba(prepared)[0]
            classes = self._pipeline.classes_
            best_idx = int(np.argmax(probs))
            return str(classes[best_idx]), float(probs[best_idx])

        label = self._pipeline.predict(prepared)[0]
        return str(label), None