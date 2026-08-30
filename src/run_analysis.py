"""Executa EDA, treinamento, avaliação e exportação dos resultados."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from modeling import QUALITY_CUTOFF, load_and_prepare, train_and_evaluate


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WineQT.csv"
RESULTS_DIR = ROOT / "results"

PT_LABELS = {
    "fixed acidity": "acidez fixa",
    "volatile acidity": "acidez volátil",
    "citric acid": "ácido cítrico",
    "residual sugar": "açúcar residual",
    "chlorides": "cloretos",
    "free sulfur dioxide": "dióxido de enxofre livre",
    "total sulfur dioxide": "dióxido de enxofre total",
    "density": "densidade",
    "pH": "pH",
    "sulphates": "sulfatos",
    "alcohol": "álcool",
    "bound sulfur dioxide": "dióxido de enxofre ligado",
    "free to total sulfur ratio": "proporção de SO₂ livre/total",
}


def create_eda() -> tuple[pd.DataFrame, pd.Series, dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    X, y, audit = load_and_prepare(DATA_PATH)
    clean = X.copy()
    clean["quality"] = raw.drop(columns=["Id"], errors="ignore").drop_duplicates()["quality"].to_numpy()
    clean["high_quality"] = y.to_numpy()

    sns.set_theme(style="whitegrid", context="notebook")
    class_counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Baixa/Média (<7)", "Alta (≥7)"], class_counts.values, color=["#d0c6bc", "#7a1f4d"])
    ax.bar_label(bars, labels=[f"{n}\n({n / len(y):.1%})" for n in class_counts.values], padding=4)
    ax.set_ylabel("Quantidade de vinhos")
    ax.set_title("A classe de alta qualidade é minoritária")
    ax.set_ylim(0, max(class_counts.values) * 1.18)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    original_features = [c for c in raw.columns if c not in ["Id", "quality"]]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, col in zip(axes.flat, original_features):
        sns.histplot(data=clean, x=col, hue="high_quality", bins=25, stat="density", common_norm=False, ax=ax, palette=["#8a817c", "#7a1f4d"], legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.flat[len(original_features):]:
        ax.axis("off")
    fig.suptitle("Distribuição das variáveis por classe de qualidade", fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    corr = clean.drop(columns=["high_quality"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlação de Pearson entre variáveis físico-químicas")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    target_corr = clean.corr(numeric_only=True)["high_quality"].drop(["high_quality", "quality"]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#3b6f8f" if value < 0 else "#7a1f4d" for value in target_corr]
    ax.barh([PT_LABELS.get(name, name) for name in target_corr.index], target_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlação com alta qualidade")
    ax.set_title("Associação individual com a classe-alvo")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "target_correlations.png", dpi=180)
    plt.close(fig)

    outlier_rows = []
    for col in original_features:
        q1, q3 = clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((clean[col] < low) | (clean[col] > high)).sum())
        outlier_rows.append({"feature": col, "lower_bound": low, "upper_bound": high, "outlier_count": count, "outlier_share": count / len(clean)})
    outliers = pd.DataFrame(outlier_rows).sort_values("outlier_share", ascending=False)
    outliers.to_csv(RESULTS_DIR / "outlier_summary.csv", index=False)

    strong_pairs = []
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    for column in upper.columns:
        for index in upper.index:
            value = upper.loc[index, column]
            if pd.notna(value) and abs(value) >= 0.50:
                strong_pairs.append({"feature_1": index, "feature_2": column, "correlation": float(value)})
    pd.DataFrame(strong_pairs).sort_values("correlation", key=abs, ascending=False).to_csv(
        RESULTS_DIR / "strong_correlations.csv", index=False
    )

    target_corr.rename("correlation").to_csv(RESULTS_DIR / "target_correlations.csv")
    eda_summary = {
        **audit,
        "quality_distribution": {str(k): int(v) for k, v in raw["quality"].value_counts().sort_index().items()},
        "target_correlations": {k: float(v) for k, v in target_corr.sort_values(key=abs, ascending=False).items()},
        "strong_feature_pairs_abs_ge_0_50": strong_pairs,
        "outlier_method": "IQR 1.5; observações preservadas por poderem representar variação química real",
        "feature_engineering": ["bound sulfur dioxide", "free to total sulfur ratio"],
    }
    with (RESULTS_DIR / "eda_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(eda_summary, stream, indent=2, ensure_ascii=False)
    return X, y, eda_summary


def main() -> None:
    X, y, eda_summary = create_eda()
    metrics, importance, best_name = train_and_evaluate(X, y, RESULTS_DIR)
    executive = {
        "dataset": eda_summary,
        "best_model": best_name,
        "best_model_test_metrics": metrics.iloc[0].to_dict(),
        "top_features": importance,
        "methodology": {
            "split": "80% treino / 20% teste, estratificado",
            "validation": "GridSearchCV com 5 folds estratificados no treino",
            "selection_metric": "average precision (PR-AUC)",
            "threshold": "otimizado por F1 usando apenas predições out-of-fold do treino",
            "test_policy": "conjunto de teste usado uma única vez na avaliação final",
        },
    }
    with (RESULTS_DIR / "executive_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(executive, stream, indent=2, ensure_ascii=False, default=float)
    print(metrics.to_string(index=False))
    print(f"\nMelhor modelo por PR-AUC na validação cruzada do treino: {best_name}")


if __name__ == "__main__":
    main()
