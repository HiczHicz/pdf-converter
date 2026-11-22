from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def audit_splits(train_data: dict, val_data: dict | None = None) -> None:
    train_X = train_data["X"]
    train_y = train_data["y"]
    train_docs = train_data["doc_ids"]

    print("=== 1. LICZNOŚCI I UNIKALNE DOKUMENTY ===")
    print(f"Train: {len(train_X)} rekordów | {len(set(train_docs))} unikalnych doc_ids")

    if val_data is not None:
        val_X = val_data["X"]
        val_y = val_data["y"]
        val_docs = val_data["doc_ids"]
        print(f"Validate: {len(val_X)} rekordów | {len(set(val_docs))} unikalnych doc_ids")

        overlap_docs = set(train_docs) & set(val_docs)
        print(f"Wspólne doc_ids (train & val): {len(overlap_docs)} -> {sorted(overlap_docs)[:10]}")

    print("\n=== 2. ROZKŁAD KLAS ===")
    print("Train klasy:", dict(Counter(train_y)))
    if val_data is not None:
        print("Validate klasy:", dict(Counter(val_y)))

    print("\n=== 3. DUPLIKATY TEKSTÓW ===")
    train_norm = [_normalize_text(t) for t in train_X]
    print(f"Train unikalne znormalizowane teksty: {len(set(train_norm))} / {len(train_norm)}")

    if val_data is not None:
        val_norm = [_normalize_text(t) for t in val_X]
        print(f"Validate unikalne znormalizowane teksty: {len(set(val_norm))} / {len(val_norm)}")

        exact_overlap = set(train_X) & set(val_X)
        norm_overlap = set(train_norm) & set(val_norm)
        print(f"Identyczne surowe teksty między train a val: {len(exact_overlap)}")
        print(f"Identyczne znormalizowane teksty między train a val: {len(norm_overlap)}")
        for sample in list(norm_overlap)[:5]:
            print("  [DUPLIKAT]:", sample[:120])


def show_top_features(fitted_classifier, top_n: int = 10) -> None:
    """Wypisuje słowa o najwyższych wagach dla każdej klasy (dla modelu z LogisticRegression)."""
    pipeline = fitted_classifier.pipeline
    features_step = pipeline.named_steps.get("features")
    clf_step = pipeline.named_steps.get("clf")

    # Obsługa zarówno pojedynczego TfidfVectorizer, jak i FeatureUnion
    if hasattr(features_step, "get_feature_names_out"):
        feature_names = features_step.get_feature_names_out()
    else:
        raise TypeError("Nie udało się pobrać nazw cech z kroku `features`.")

    # Jeśli clf to VotingClassifier, wyciągamy estymator 'lr'
    if hasattr(clf_step, "coef_"):
        lr_model = clf_step
    elif hasattr(clf_step, "named_estimators_") and "lr" in clf_step.named_estimators_:
        lr_model = clf_step.named_estimators_["lr"]
    else:
        print("Ten klasyfikator nie posiada bezpośrednich wag `.coef_`.")
        return

    classes = lr_model.classes_
    coefs = lr_model.coef_

    print(f"\n=== 4. TOP {top_n} SŁÓW DLA KAŻDEJ KLASY ===")
    if coefs.shape[0] == 1 and len(classes) == 2:
        # Klasyfikacja binarna w sklearn ma jeden wiersz współczynników dla classes[1]
        weights = coefs[0]
        top_pos = weights.argsort()[::-1][:top_n]
        top_neg = weights.argsort()[:top_n]
        print(f"Klasa '{classes[1]}' (+): " + ", ".join(f"{feature_names[i]} ({weights[i]:.2f})" for i in top_pos))
        print(f"Klasa '{classes[0]}' (-): " + ", ".join(f"{feature_names[i]} ({weights[i]:.2f})" for i in top_neg))
    else:
        for idx, cls_name in enumerate(classes):
            weights = coefs[idx]
            top_idx = weights.argsort()[::-1][:top_n]
            top_str = ", ".join(f"{feature_names[i]} ({weights[i]:.2f})" for i in top_idx)
            print(f"Klasa '{cls_name}': {top_str}")



def evaluate_confidence_thresholds(
    fitted_classifier,
    X: list,
    y_true: list,
    thresholds: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """
    Dla każdego progu pewności (max_proba >= t) liczy:
    - coverage_pct: % danych spełniających próg,
    - accuracy_pct: % poprawnych trafień wśród tych danych,
    - f1_macro_pct: F1 macro (%) na tym podzbiorze.
    """
    probas = fitted_classifier.predict_proba(X)
    classes = list(fitted_classifier.pipeline.named_steps["clf"].classes_)

    max_conf = np.max(probas, axis=1)
    pred_idx = np.argmax(probas, axis=1)
    y_pred = np.array([classes[i] for i in pred_idx])
    y_arr = np.asarray(y_true)

    total_n = len(y_arr)
    rows = []

    for t in thresholds:
        mask = max_conf >= t
        n_selected = int(np.sum(mask))
        coverage_pct = (n_selected / total_n * 100.0) if total_n > 0 else 0.0

        if n_selected == 0:
            rows.append({
                "threshold": t,
                "n_samples": 0,
                "coverage_pct": 0.0,
                "accuracy_pct": np.nan,
                "f1_macro_pct": np.nan,
            })
            continue

        y_true_sub = y_arr[mask]
        y_pred_sub = y_pred[mask]

        acc_pct = float(np.mean(y_true_sub == y_pred_sub) * 100.0)
        f1_pct = float(f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0) * 100.0)

        rows.append({
            "threshold": t,
            "n_samples": n_selected,
            "coverage_pct": round(coverage_pct, 2),
            "accuracy_pct": round(acc_pct, 2),
            "f1_macro_pct": round(f1_pct, 2),
        })

    return pd.DataFrame(rows)