import pandas as pd
import numpy as np


def compute_stats(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(include=["object", "category"])

    column_stats = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        column_stats[col] = {
            "type": "numeric",
            "count": int(s.count()),
            "missing": int(numeric[col].isna().sum()),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q25": round(float(s.quantile(0.25)), 4),
            "q75": round(float(s.quantile(0.75)), 4),
        }

    for col in categorical.columns:
        s = categorical[col].dropna()
        vc = s.value_counts()
        column_stats[col] = {
            "type": "categorical",
            "count": int(s.count()),
            "missing": int(categorical[col].isna().sum()),
            "unique": int(s.nunique()),
            "top_values": vc.head(5).to_dict(),
        }

    # Correlations (numeric only, if >= 2 columns)
    correlations = {}
    if len(numeric.columns) >= 2:
        corr_matrix = numeric.corr()
        pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(corr_matrix.iloc[i, j]), 4),
                })
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        correlations["top_pairs"] = pairs[:10]

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "numeric_columns": list(numeric.columns),
        "categorical_columns": list(categorical.columns),
        "missing_values": df.isna().sum().to_dict(),
        "column_stats": column_stats,
        "correlations": correlations,
    }
