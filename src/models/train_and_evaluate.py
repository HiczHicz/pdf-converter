from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


@dataclass
class ClassifierConfig:
    """Parametry strojenia przekazywane bezpośrednio z kodu."""
    model_type: Literal["simple", "ensemble"] = "simple"

    # Wspólne parametry przetwarzania tekstu i TF-IDF
    boost_intro_words: int = 100
    max_word_features: int = 2_500
    word_ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 4
    max_df: float = 0.85
    sublinear_tf: bool = True

    # Regularyzacja i wagi dla modeli liniowych / NB
    C: float = 0.25
    nb_alpha: float = 1.0
    class_weight: Literal["balanced"] | None = "balanced"

    # Parametry dodatkowe używane przy model_type="ensemble"
    use_char_features: bool = False
    max_char_features: int = 4_000
    char_ngram_range: tuple[int, int] = (3, 4)
    char_min_df: int = 5
    calibration_cv: int = 3
    ensemble_weights: list[float] = field(default_factory=lambda: [2.0, 1.5, 1.0])

    random_state: int = 42


class TextClassifier:
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.is_fitted = False
        self.pipeline = self._build_pipeline(self.config)

    def _build_features(self, cfg: ClassifierConfig):
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=cfg.word_ngram_range,
            max_features=cfg.max_word_features,
            sublinear_tf=cfg.sublinear_tf,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
        )

        if cfg.model_type == "simple" or not cfg.use_char_features:
            return word_vec

        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=cfg.char_ngram_range,
            max_features=cfg.max_char_features,
            sublinear_tf=cfg.sublinear_tf,
            min_df=cfg.char_min_df,
            max_df=cfg.max_df,
        )
        return FeatureUnion([
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec),
        ])

    def _build_estimator(self, cfg: ClassifierConfig):
        if cfg.model_type == "simple":
            return LogisticRegression(
                C=cfg.C,
                max_iter=1000,
                class_weight=cfg.class_weight,
                random_state=cfg.random_state,
            )

        if cfg.model_type == "ensemble":
            cv_cal = StratifiedKFold(
                n_splits=cfg.calibration_cv,
                shuffle=True,
                random_state=cfg.random_state,
            )
            lr = LogisticRegression(
                C=cfg.C,
                max_iter=1000,
                class_weight=cfg.class_weight,
                random_state=cfg.random_state,
            )
            svc = CalibratedClassifierCV(
                LinearSVC(
                    C=cfg.C * 0.5,
                    class_weight=cfg.class_weight,
                    random_state=cfg.random_state,
                ),
                cv=cv_cal,
            )
            cnb = ComplementNB(alpha=cfg.nb_alpha)

            return VotingClassifier(
                estimators=[("lr", lr), ("svc", svc), ("cnb", cnb)],
                voting="soft",
                weights=cfg.ensemble_weights,
            )

        raise ValueError(f"Nieznany model_type: {cfg.model_type}")

    def _build_pipeline(self, cfg: ClassifierConfig) -> Pipeline:
        return Pipeline([
            ("features", self._build_features(cfg)),
            ("clf", self._build_estimator(cfg)),
        ])

    def _prepare_texts(self, texts: list[str] | list[list[str]]) -> list[str]:
        """Skleja tokeny (jeśli podano listy) i opcjonalnie wzmacnia początek dokumentu."""
        if not texts:
            return []

        boost = self.config.boost_intro_words
        prepared: list[str] = []
        for item in texts:
            text = " ".join(item) if isinstance(item, list) else item
            if boost > 0:
                words = text.split()
                intro = " ".join(words[:boost])
                text = f"{intro} {text}"
            prepared.append(text)
        return prepared

    def evaluate_cv(
        self,
        X: list[str] | list[list[str]],
        y: list,
        groups: list | np.ndarray | None = None,
        n_splits: int = 5,
        average: Literal["macro", "weighted", "micro"] = "macro",
    ) -> dict[str, float | list[float] | str]:
        """
        Ocenia model w kros-walidacji, pilnując aby ten sam dokument/grupa
        nie trafił jednocześnie do train i test w danym foldzie.
        Zwraca średnią miar testowych po wszystkich foldach.
        """
        X_clean = self._prepare_texts(X)
        X_arr = np.asarray(X_clean, dtype=object)
        y_arr = np.asarray(y)

        if len(X_arr) != len(y_arr):
            raise ValueError("Długość `X` i `y` musi być taka sama.")

        if groups is not None:
            groups_arr = np.asarray(groups)
            if len(groups_arr) != len(y_arr):
                raise ValueError("Długość `groups` musi być równa długości `y`.")

            unique_groups = np.unique(groups_arr)
            if len(unique_groups) < n_splits:
                raise ValueError(
                    f"Liczba unikalnych grup ({len(unique_groups)}) jest mniejsza niż n_splits ({n_splits})."
                )

            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )
            splits = cv.split(X_arr, y_arr, groups=groups_arr)
        else:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )
            splits = cv.split(X_arr, y_arr)

        f1_macro_folds: list[float] = []
        f1_weighted_folds: list[float] = []
        f1_selected_folds: list[float] = []
        all_y_true: list = []
        all_y_pred: list = []

        for train_idx, test_idx in splits:
            if groups is not None:
                train_groups = set(groups_arr[train_idx].tolist())
                test_groups = set(groups_arr[test_idx].tolist())
                overlap = train_groups & test_groups
                assert not overlap, f"Wyciek grup między train i test: {overlap}"

            fold_model = clone(self.pipeline)
            fold_model.fit(X_arr[train_idx].tolist(), y_arr[train_idx])
            y_test_pred = fold_model.predict(X_arr[test_idx].tolist())

            f1_macro_folds.append(
                float(f1_score(y_arr[test_idx], y_test_pred, average="macro", zero_division=0))
            )
            f1_weighted_folds.append(
                float(f1_score(y_arr[test_idx], y_test_pred, average="weighted", zero_division=0))
            )
            f1_selected_folds.append(
                float(f1_score(y_arr[test_idx], y_test_pred, average=average, zero_division=0))
            )

            all_y_true.extend(y_arr[test_idx].tolist())
            all_y_pred.extend(y_test_pred.tolist())

        return {
            "f1_macro": float(np.mean(f1_macro_folds)),
            "f1_macro_std": float(np.std(f1_macro_folds)),
            "f1_macro_folds": f1_macro_folds,
            "f1_weighted": float(np.mean(f1_weighted_folds)),
            "f1_weighted_folds": f1_weighted_folds,
            "f1_selected": float(np.mean(f1_selected_folds)),
            "f1_selected_folds": f1_selected_folds,
            "report": classification_report(all_y_true, all_y_pred, zero_division=0),
        }

    def fit(self, X: list[str] | list[list[str]], y: list) -> "TextClassifier":
        """Trenuje finalny model na całym przekazanym zbiorze danych."""
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
                "config": self.config,
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
        config = data.get("config") or ClassifierConfig()
        instance = cls(config=config)
        instance.pipeline = data["pipeline"]
        instance.is_fitted = True
        return instance