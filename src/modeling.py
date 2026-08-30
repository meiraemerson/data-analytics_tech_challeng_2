"""Pipeline reprodutível para classificação binária da qualidade de vinhos."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TARGET = "high_quality"
QUALITY_CUTOFF = 7


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos químicos interpretáveis sem utilizar a variável alvo."""
    data = frame.copy()
    data["bound sulfur dioxide"] = (
        data["total sulfur dioxide"] - data["free sulfur dioxide"]
    ).clip(lower=0)
    data["free to total sulfur ratio"] = data["free sulfur dioxide"] / data[
        "total sulfur dioxide"
    ].replace(0, np.nan)
    data["free to total sulfur ratio"] = data["free to total sulfur ratio"].fillna(0)
    return data


def load_and_prepare(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Carrega a base, remove identificador e duplicatas e cria a classe binária."""
    raw = pd.read_csv(data_path)
    original_rows = len(raw)
    missing_values = int(raw.isna().sum().sum())
    id_unique = bool(raw["Id"].is_unique) if "Id" in raw else None

    modeling = raw.drop(columns=["Id"], errors="ignore")
    duplicates = int(modeling.duplicated().sum())
    modeling = modeling.drop_duplicates().reset_index(drop=True)
    y = (modeling["quality"] >= QUALITY_CUTOFF).astype(int).rename(TARGET)
    X = add_features(modeling.drop(columns=["quality"]))

    audit = {
        "original_rows": original_rows,
        "rows_after_duplicate_removal": len(modeling),
        "duplicates_removed": duplicates,
        "missing_values": missing_values,
        "id_was_unique": id_unique,
        "positive_class_count": int(y.sum()),
        "positive_class_share": float(y.mean()),
        "negative_class_count": int((1 - y).sum()),
        "target_rule": f"quality >= {QUALITY_CUTOFF}",
    }
    return X, y, audit


def _model_spaces(numeric_features: list[str]) -> dict:
    scaled_preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), numeric_features)], remainder="drop"
    )
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("preprocess", scaled_preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.05, 0.2, 1.0, 5.0, 20.0]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced_subsample",
                            n_jobs=1,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 3, 6],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "SVM RBF": (
            Pipeline(
                [
                    ("preprocess", clone(scaled_preprocessor)),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.25, 1.0, 4.0, 12.0], "model__gamma": ["scale", 0.03, 0.1, 0.3]},
        ),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    results_dir: str | Path,
) -> tuple[pd.DataFrame, dict, str]:
    """Ajusta três modelos, seleciona limiar sem tocar no teste e salva artefatos."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    rows: list[dict] = []
    curves: dict[str, dict] = {}

    for name, (pipeline, grid) in _model_spaces(list(X.columns)).items():
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        oof_prob = cross_val_predict(
            clone(best), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        threshold = _best_f1_threshold(y_train, oof_prob)
        test_prob = best.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        metrics = {
            "model": name,
            "cv_pr_auc": float(search.best_score_),
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, test_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred, zero_division=0),
            "f1": f1_score(y_test, test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_prob),
            "pr_auc": average_precision_score(y_test, test_prob),
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        }
        rows.append(metrics)
        fitted[name] = best
        thresholds[name] = threshold
        curves[name] = {"prob": test_prob, "pred": test_pred}

        report = classification_report(
            y_test,
            test_pred,
            target_names=["Baixa/Média", "Alta"],
            output_dict=True,
            zero_division=0,
        )
        with (results_dir / f"classification_report_{name.lower().replace(' ', '_')}.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, test_pred),
            display_labels=["Baixa/Média", "Alta"],
        ).plot(ax=ax, cmap="Purples", colorbar=False)
        ax.set_title(f"Matriz de confusão — {name}")
        fig.tight_layout()
        fig.savefig(results_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=180)
        plt.close(fig)

    # A escolha do vencedor usa somente a validação cruzada do treino.
    # As métricas de teste permanecem uma estimativa final, não um critério de seleção.
    metrics_df = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(results_dir / "model_metrics.csv", index=False)
    best_name = str(metrics_df.iloc[0]["model"])
    bundle = {
        "model": fitted[best_name],
        "threshold": thresholds[best_name],
        "features": list(X.columns),
        "target_rule": f"quality >= {QUALITY_CUTOFF}",
        "model_name": best_name,
    }
    joblib.dump(bundle, results_dir / "best_model.joblib")

    prediction_table = X_test.copy()
    prediction_table["actual_high_quality"] = y_test.values
    prediction_table["predicted_probability"] = curves[best_name]["prob"]
    prediction_table["predicted_high_quality"] = curves[best_name]["pred"]
    prediction_table.to_csv(results_dir / "test_predictions.csv", index=False)

    _plot_model_comparison(metrics_df, results_dir)
    _plot_curves(curves, y_test, results_dir)
    importance = _plot_importance(fitted[best_name], X_test, y_test, results_dir)
    return metrics_df, importance, best_name


def _plot_model_comparison(metrics_df: pd.DataFrame, results_dir: Path) -> None:
    long = metrics_df.melt(
        id_vars="model",
        value_vars=["pr_auc", "roc_auc", "f1", "recall", "precision", "balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax, palette="Set2")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Resultado no conjunto de teste")
    ax.set_title("Comparação dos modelos — várias métricas")
    ax.legend(title="Modelo", loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=180)
    plt.close(fig)


def _plot_curves(curves: dict, y_test: pd.Series, results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, values in curves.items():
        fpr, tpr, _ = roc_curve(y_test, values["prob"])
        precision, recall, _ = precision_recall_curve(y_test, values["prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, values['prob']):.3f})")
        axes[1].plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, values['prob']):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(title="Curva ROC", xlabel="Taxa de falso positivo", ylabel="Taxa de verdadeiro positivo")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", linewidth=1, label="Base aleatória")
    axes[1].set(title="Curva Precisão–Recall", xlabel="Recall", ylabel="Precisão")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "roc_pr_curves.png", dpi=180)
    plt.close(fig)


def _plot_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, results_dir: Path) -> dict:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    top = importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#7a1f4d")
    ax.set_xlabel("Queda média no PR-AUC ao embaralhar a variável")
    ax.set_title("Variáveis mais influentes — importância por permutação")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=180)
    plt.close(fig)
    return importance.head(10).to_dict(orient="records")
