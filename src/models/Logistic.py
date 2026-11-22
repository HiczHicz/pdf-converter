from pathlib import Path
from typing import Literal
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


class TextClassifier:
    def __init__(
        self,
        multilabel: bool = False,
        max_features: int = 25_000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.multilabel = multilabel
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer() if multilabel else None
        self.is_fitted = False

        base_clf = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        clf = OneVsRestClassifier(base_clf, n_jobs=-1) if multilabel else base_clf

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True,
                min_df=2,
                max_df=0.9,
            )),
            ("clf", clf),
        ])

    @staticmethod
    def _prepare_texts(texts: list[str] | list[list[str]]) -> list[str]:
        """Akceptuje listę stringów lub listę list tokenów (po NLTK) i zwraca list[str]."""
        if not texts:
            return []
        if isinstance(texts[0], list):
            return [" ".join(tokens) for tokens in texts]
        return texts  # type: ignore[return-value]

    def fit(self, X: list[str] | list[list[str]], y: list) -> "TextClassifier":
        """Trenuje model na pełnym zbiorze danych."""
        X_clean = self._prepare_texts(X)
        y_encoded = self.mlb.fit_transform(y) if self.multilabel and self.mlb else y

        self.pipeline.fit(X_clean, y_encoded)
        self.is_fitted = True
        return self

    def evaluate(
        self,
        X: list[str] | list[list[str]],
        y: list,
        test_size: float = 0.2,
        average: Literal["macro", "weighted", "micro"] = "macro",
    ) -> dict[str, float | str]:
        """Dzieli dane na train/test, trenuje model i zwraca metryki F1 oraz raport."""
        X_clean = self._prepare_texts(X)

        if self.multilabel and self.mlb:
            y_encoded = self.mlb.fit_transform(y)
            stratify = None
        else:
            y_encoded = y
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean,
            y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        y_pred = self.pipeline.predict(X_test)

        target_names = [str(c) for c in self.mlb.classes_] if self.multilabel and self.mlb else None

        return {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_selected": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
            "report": classification_report(y_test, y_pred, target_names=target_names, zero_division=0),
        }

    def predict(self, X: list[str] | list[list[str]]) -> list:
        """Zwraca przewidziane etykiety dla nowych tekstów."""
        if not self.is_fitted:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj najpierw fit() lub evaluate().")

        X_clean = self._prepare_texts(X)
        preds = self.pipeline.predict(X_clean)

        if self.multilabel and self.mlb:
            return [list(labels) for labels in self.mlb.inverse_transform(preds)]
        return preds.tolist()

    def predict_proba(self, X: list[str] | list[list[str]]):
        """Zwraca macierz prawdopodobieństw dla każdej klasy (przydatne do strojenia progu pod F1)."""
        if not self.is_fitted:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj najpierw fit() lub evaluate().")

        X_clean = self._prepare_texts(X)
        return self.pipeline.predict_proba(X_clean)

    def save(self, model_path: str | Path = "./models/text_classifier.joblib") -> Path:
        """Zapisuje wytrenowany model na dysk."""
        if not self.is_fitted:
            raise RuntimeError("Nie można zapisać niewytrenowanego modelu.")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "multilabel": self.multilabel,
                "mlb": self.mlb,
            },
            path,
            compress=3,
        )
        return path

    @classmethod
    def load(cls, model_path: str | Path = "./models/text_classifier.joblib") -> "TextClassifier":
        """Wczytuje wytrenowany model z dysku."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Plik modelu nie istnieje: {path}")

        data = joblib.load(path)
        instance = cls(multilabel=data["multilabel"])
        instance.pipeline = data["pipeline"]
        instance.mlb = data["mlb"]
        instance.is_fitted = True
        return instance