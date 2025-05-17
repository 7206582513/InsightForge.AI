import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from modules.plot_utils import generate_charts
from modules.pdf_generator import generate_pdf_from_charts


def knn_impute(df, n_neighbors=5):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_encoded = df.copy()
    if cat_cols:
        df_encoded[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_array = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(df_array, columns=df.columns)
    if cat_cols:
        df_imputed[cat_cols] = df_imputed[cat_cols].round().astype(int)
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
    return df_imputed


def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            temp = df[col].astype(str)
            temp = temp.str.replace(r"[₹,kKmMsS\sa-zA-Z:]+", "", regex=True)
            numeric = pd.to_numeric(temp, errors='coerce')
            if numeric.notnull().mean() > 0.6:
                df[col] = numeric
    return df


def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data[vif_data['VIF'] > 5]['feature'].tolist()


def auto_eda_pipeline(df, task_type="classification", target_col=None):
    report = {}

    # Step 1: Clean column names
    df.columns = df.columns.str.strip()

    # Step 2: Drop high-null and ID-like columns
    df = df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'number' in col.lower()], errors='ignore')
    df = df.dropna(axis=1, thresh=int(0.6 * len(df)))

    # Step 3: Convert dirty strings to numbers where possible
    df = clean_numeric_columns(df)

    # Step 4: Impute missing values
    df = knn_impute(df)

    # Step 5: Select target column
    if not target_col or target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataset.")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Step 6: Encode features
    for col in X.select_dtypes(include='object').columns:
        if X[col].nunique() <= 10:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X[col] = LabelEncoder().fit_transform(X[col])

    # Step 7: Encode target if classification
    if task_type == "classification" and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Step 8: Standard Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Step 9: Chi2 feature removal (classification only)
    chi2_removed = []
    if task_type == "classification":
        for col in X_scaled.select_dtypes(include='float64').columns:
            try:
                stat, p = chi2(X_scaled[[col]], y)
                if p[0] > 0.05:
                    chi2_removed.append(col)
                    X_scaled.drop(columns=[col], inplace=True)
            except:
                continue

    # Step 10: VIF-based multicollinearity removal
    vif_cols = calculate_vif(X_scaled)
    X_scaled.drop(columns=vif_cols, inplace=True)

    # Step 11: Outlier removal using IQR
    Q1 = X_scaled.quantile(0.25)
    Q3 = X_scaled.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = ~((X_scaled < lower) | (X_scaled > upper)).any(axis=1)
    X_clean = X_scaled[mask]
    y_clean = y.loc[mask]

    # Final cleaned DataFrame
    final_df = pd.concat([X_clean.reset_index(drop=True), y_clean.reset_index(drop=True)], axis=1)

    # 🚨 Final NaN removal
    final_df = final_df.dropna()

    # Step 12: Generate charts and PDF
    charts = generate_charts(final_df)
    pdf_path = generate_pdf_from_charts(charts)

    # Report summary
    report['task'] = task_type
    report['target'] = target_col
    report['shape_before'] = df.shape
    report['shape_after'] = final_df.shape
    report['outliers_removed'] = int(df.shape[0] - final_df.shape[0])
    report['chi2_removed'] = chi2_removed
    report['vif_removed'] = vif_cols
    report['charts'] = charts
    report['pdf'] = pdf_path

    return final_df, report