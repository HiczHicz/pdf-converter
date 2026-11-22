from pathlib import Path
from typing import Literal
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


class TextClassifier:
    def __init__(
        self,
        max_word_features: int = 6_000,
        max_char_features: int = 10_000,
        C: float = 0.5,
        boost_intro_words: int = 150,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.boost_intro_words = boost_intro_words
        self.is_fitted = False

        # Mniejszy, mocniej odfiltrowany słownik (min_df=3) zapobiega przeuczeniu na 240 plikach
        features = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=max_word_features,
                sublinear_tf=True,
                min_df=3,
                max_df=0.85,
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_char_features,
                sublinear_tf=True,
                min_df=4,
                max_df=0.85,
            )),
        ])

        lr = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        svc = CalibratedClassifierCV(
            LinearSVC(C=C * 0.5, class_weight="balanced", random_state=random_state),
            cv=3,
        )
        cnb = ComplementNB(alpha=0.5)

        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("svc", svc), ("cnb", cnb)],
            voting="soft",
            weights=[2.0, 2.0, 1.5],
        )

        self.pipeline = Pipeline([
            ("features", features),
            ("clf", ensemble),
        ])

    def _prepare_texts(self, texts: list[str] | list[list[str]]) -> list[str]:
        """Skleja tokeny (jeśli podano listy) i wzmacnia pierwsze N słów dokumentu."""
        if not texts:
            return []

        prepared: list[str] = []
        for item in texts:
            text = " ".join(item) if isinstance(item, list) else item
            if self.boost_intro_words > 0:
                words = text.split()
                intro = " ".join(words[: self.boost_intro_words])
                text = f"{intro} {text}"
            prepared.append(text)
        return prepared

    def evaluate_cv(
        self,
        X: list[str] | list[list[str]],
        y: list,
        n_splits: int = 5,
        average: Literal["macro", "weighted", "micro"] = "macro",
    ) -> dict[str, float | str]:
        """
        Ocenia model za pomocą 5-krotnej walidacji krzyżowej (StratifiedKFold)
        na wszystkich 240 dokumentach bez utraty danych treningowych.
        """
        X_clean = self._prepare_texts(X)
        y_arr = np.array(y)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        y_pred = cross_val_predict(self.pipeline, X_clean, y_arr, cv=cv, n_jobs=-1)

        return {
            "f1_macro": float(f1_score(y_arr, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_arr, y_pred, average="weighted", zero_division=0)),
            "f1_selected": float(f1_score(y_arr, y_pred, average=average, zero_division=0)),
            "report": classification_report(y_arr, y_pred, zero_division=0),
        }

    def fit(self, X: list[str] | list[list[str]], y: list) -> "TextClassifier":
        """Trenuje finalny model na 100% dostępnych danych (wszystkich 240 dokumentach)."""
        X_clean = self._prepare_texts(X)
        self.pipeline.fit(X_clean, np.array(y))
        self.is_fitted = True
        return self

    def predict(self, X: list[str] | list[list[str]]) -> list:
        if not self.is_fitted:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj najpierw fit().")
        X_clean = self._prepare_texts(X)
        return self.pipeline.predict(X_clean).tolist()

    def predict_proba(self, X: list[str] | list[list[str]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj najpierw fit().")
        X_clean = self._prepare_texts(X)
        return self.pipeline.predict_proba(X_clean)

    def save(self, model_path: str | Path = "./models/text_classifier.joblib") -> Path:
        if not self.is_fitted:
            raise RuntimeError("Nie można zapisać niewytrenowanego modelu.")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "boost_intro_words": self.boost_intro_words,
            },
            path,
            compress=3,
        )
        return path

    @classmethod
    def load(cls, model_path: str | Path = "./models/text_classifier.joblib") -> "TextClassifier":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Plik modelu nie istnieje: {path}")
        data = joblib.load(path)
        instance = cls(boost_intro_words=data.get("boost_intro_words", 150))
        instance.pipeline = data["pipeline"]
        instance.is_fitted = True
        return instance