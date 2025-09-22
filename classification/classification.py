import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import numpy as np
import copy
import os


# 1. Load data and create grading labels (No modification needed)
def load_and_prepare(path):
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    except FileNotFoundError:
        print(f"Error: File '{path}' not found. Please check the path.")
        return None
    bins = [float("-inf"), 1.0, 2.0, 3.0, float("inf")]
    labels = [1, 2, 3, 4]
    if "左柱镜" not in df.columns or "右柱镜" not in df.columns:
        print("Error: Missing '左柱镜' or '右柱镜' column in the file.")
        return None
    df["left_grade"] = pd.cut(
        df["左柱镜"], bins=bins, labels=labels, include_lowest=True
    )
    df["right_grade"] = pd.cut(
        df["右柱镜"], bins=bins, labels=labels, include_lowest=True
    )
    df = df.dropna(subset=["left_grade", "right_grade"])
    df["left_grade"] = df["left_grade"].astype(int)
    df["right_grade"] = df["right_grade"].astype(int)
    return df


# 2. Cross-validation, performance evaluation and feature importance analysis (Major modification)
def cross_validate_model(
    df, target_col, numeric_features, categorical_features, cv=5
):  # Function renamed
    from sklearn.exceptions import FitFailedWarning
    import warnings

    warnings.simplefilter("ignore", FitFailedWarning)

    X = df.drop(
        columns=["左球镜", "右球镜", "左柱镜", "右柱镜", "left_grade", "right_grade"]
    )
    y_raw = df[target_col]
    y = (y_raw - 1).astype(int)
    all_features = numeric_features + categorical_features
    X = X[all_features]

    print(f"Performing {cv}-fold cross-validation for target: {target_col}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features used: {len(all_features)}")

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    preprocessor_for_names = copy.deepcopy(preprocessor)
    preprocessor_for_names.fit(X)
    all_feature_names_out = preprocessor_for_names.get_feature_names_out()
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "") for name in all_feature_names_out
    ]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred_all, y_true_all, acc_scores = [], [], []
    feature_importances = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=4,
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)
        acc_scores.append((y_pred == y_test).mean())

        xgb_model = pipeline.named_steps["clf"]
        current_feature_names = pipeline.named_steps["prep"].get_feature_names_out()
        fold_importance = pd.Series(
            xgb_model.feature_importances_, index=current_feature_names
        )
        fold_importance = fold_importance.reindex(all_feature_names_out, fill_value=0)
        feature_importances.append(fold_importance.values)

    y_pred_all, y_true_all = pd.Series(y_pred_all) + 1, pd.Series(y_true_all) + 1

    print(f"\nMean accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print("Classification report:")
    print(classification_report(y_true_all, y_pred_all, digits=4))

    report = classification_report(y_true_all, y_pred_all, output_dict=True)
    metrics = {
        "accuracy": np.mean(acc_scores),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
    }

    avg_importance = pd.DataFrame(
        feature_importances, columns=clean_feature_names
    ).mean()
    aggregated_importance = {}
    for num_feat in numeric_features:
        if num_feat in avg_importance.index:
            aggregated_importance[num_feat] = avg_importance[num_feat]
    for cat_feat in categorical_features:
        pattern = f"{cat_feat}_"
        cat_columns = [col for col in avg_importance.index if col.startswith(pattern)]
        if cat_columns:
            aggregated_importance[cat_feat] = avg_importance[cat_columns].sum()
    aggregated_series = pd.Series(aggregated_importance).sort_values(ascending=False)

    return metrics, aggregated_series


# 3. Calculate composite feature importance (No modification needed)
def calculate_composite_importance(left_agg, right_agg, output_dir, scenario_name):
    if left_agg is None or right_agg is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    composite = pd.DataFrame(
        {"left_eye_importance": left_agg, "right_eye_importance": right_agg}
    )
    composite["composite_importance"] = composite.mean(axis=1)
    composite = composite.sort_values(by="composite_importance", ascending=False)
    composite_path = os.path.join(
        output_dir, f"{scenario_name}_composite_importance.csv"
    )
    composite.to_csv(composite_path, encoding="utf-8-sig")
    return composite


# Main workflow
if __name__ == "__main__":
    # --- 1. Configuration ---
    data_path = r"./data_new.csv"
    base_numeric_feats = [
        "年龄",
        "身高",
        "体重",
        "父球镜",
        "父柱镜",
        "母球镜",
        "母柱镜",
        "出生体重",
    ]
    base_categorical_feats = [
        "性别",
        "分娩方式",
        "出生史",
        "出生吸氧史",
        "居住地",
        "户外活动时间",
        "近距离用眼时间",
        "电子产品使用时间",
        "营养是否均衡",
    ]
    new_features = [
        "左眼睫毛下垂角度",
        "右眼睫毛下垂角度",
        "下睑缘相对距离",
        "外眦点相对距离",
        "角膜中心相对距离",
        "下睑缘与内眦连线夹角",
        "角膜中心与内眦连线夹角",
    ]
    output_dir = "model_comparison_results"  # Changed output directory for distinction

    # --- 2. Data loading ---
    df_original = load_and_prepare(data_path)

    if df_original is not None:
        # --- 3. Scenario 1: Modeling and analysis with basic features ---
        print("\n" + "=" * 80)
        print(" Scenario 1: Modeling with basic features ".center(80, "="))
        print("=" * 80 + "\n")
        results_dir_base = os.path.join(output_dir, "base_model_results")

        # Call the renamed function
        metrics_left_base, imp_left_base = cross_validate_model(
            df_original, "left_grade", base_numeric_feats, base_categorical_feats, cv=5
        )
        os.makedirs(results_dir_base, exist_ok=True)
        imp_left_base.to_csv(
            os.path.join(results_dir_base, "base_left_eye_importance.csv"),
            encoding="utf-8-sig",
        )
        metrics_right_base, imp_right_base = cross_validate_model(
            df_original, "right_grade", base_numeric_feats, base_categorical_feats, cv=5
        )
        imp_right_base.to_csv(
            os.path.join(results_dir_base, "base_right_eye_importance.csv"),
            encoding="utf-8-sig",
        )
        comp_imp_base = calculate_composite_importance(
            imp_left_base, imp_right_base, results_dir_base, "base"
        )
        print(f"\nBasic model feature importance report saved to: {results_dir_base}")

        # --- 4. Scenario 2: Modeling and analysis with extended features ---
        print("\n" + "=" * 80)
        print(" Scenario 2: Modeling with extended features ".center(80, "="))
        print(" (Rows with any missing values have been removed) ".center(80, " "))
        print("=" * 80 + "\n")
        results_dir_extended = os.path.join(output_dir, "extended_model_results")
        os.makedirs(results_dir_extended, exist_ok=True)
        extended_numeric_feats = base_numeric_feats + new_features
        all_model_features = (
            extended_numeric_feats
            + base_categorical_feats
            + ["左球镜", "右球镜", "左柱镜", "右柱镜", "left_grade", "right_grade"]
        )
        rows_before = len(df_original)
        df_extended_clean = df_original.dropna(subset=all_model_features)
        rows_after = len(df_extended_clean)
        print(
            f"Data cleaning: Original rows {rows_before}, remaining rows after removing missing values {rows_after} (removed {rows_before - rows_after} rows).\n"
        )
        metrics_left_extended, metrics_right_extended = {}, {}
        imp_left_extended, imp_right_extended, comp_imp_extended = None, None, None
        if not df_extended_clean.empty and len(df_extended_clean) > 20:
            metrics_left_extended, imp_left_extended = cross_validate_model(
                df_extended_clean,
                "left_grade",
                extended_numeric_feats,
                base_categorical_feats,
                cv=5,
            )
            imp_left_extended.to_csv(
                os.path.join(results_dir_extended, "extended_left_eye_importance.csv"),
                encoding="utf-8-sig",
            )
            metrics_right_extended, imp_right_extended = cross_validate_model(
                df_extended_clean,
                "right_grade",
                extended_numeric_feats,
                base_categorical_feats,
                cv=5,
            )
            imp_right_extended.to_csv(
                os.path.join(results_dir_extended, "extended_right_eye_importance.csv"),
                encoding="utf-8-sig",
            )
            comp_imp_extended = calculate_composite_importance(
                imp_left_extended, imp_right_extended, results_dir_extended, "extended"
            )
            print(
                f"\nExtended model feature importance report saved to: {results_dir_extended}"
            )
        else:
            print(
                "Cleaned data is empty or too few samples, skipping extended feature modeling."
            )

        # --- 5. Final summary of results ---
        avg_metrics_base = {
            m: (metrics_left_base.get(m, 0) + metrics_right_base.get(m, 0)) / 2
            for m in metrics_left_base
        }
        avg_metrics_extended = {
            m: (metrics_left_extended.get(m, 0) + metrics_right_extended.get(m, 0)) / 2
            for m in metrics_left_extended
        }
        print("\n" + "=" * 80)
        print(
            " Summary of average performance for left and right eyes ".center(80, "=")
        )
        print("=" * 80 + "\n")
        summary_data = []
        metric_names_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
        }
        for metric_key, metric_display_name in metric_names_map.items():
            summary_data.append(
                {
                    "Metric": metric_display_name,
                    "Basic Feature Model (Average)": f"{avg_metrics_base.get(metric_key, 0):.4f}",
                    "Extended Feature Model (Average)": f"{avg_metrics_extended.get(metric_key, 0):.4f}",
                }
            )
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print("\n" + "-" * 80)
        print(
            " Improvement of metrics (Extended features vs Basic features) ".center(
                80, "-"
            )
        )
        print("-" * 80)
        for metric_key, metric_display_name in metric_names_map.items():
            improvement = avg_metrics_extended.get(
                metric_key, 0
            ) - avg_metrics_base.get(metric_key, 0)
            print(f"{metric_display_name} improvement: {improvement:+.4f}")
        print("\n" + "=" * 80)
        print(" Quick preview of main influencing factors ".center(80, "="))
        print("=" * 80 + "\n")
        if comp_imp_base is not None:
            print("--- Scenario 1 (Basic features) Top 10 influencing factors ---")
            print(comp_imp_base["composite_importance"].head(10))
        if comp_imp_extended is not None:
            print("\n--- Scenario 2 (Extended features) Top 10 influencing factors ---")
            print(comp_imp_extended["composite_importance"].head(10))
