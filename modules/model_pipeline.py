import os
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("static/charts", exist_ok=True)
    path = f"static/charts/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_regression(y_true, y_pred, model_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.title(f"Regression Plot - {model_name}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    os.makedirs("static/charts", exist_ok=True)
    path = f"static/charts/regression_plot_{model_name.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    return path

def train_best_model(df, task_type="classification"):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    is_classification = task_type == "classification"

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle imbalance
    if is_classification:
        class_counts = Counter(y_train)
        min_class = min(class_counts.values())

        if min_class >= 6:
            sampler = SMOTE(k_neighbors=5, random_state=42)
        elif min_class > 1:
            sampler = SMOTE(k_neighbors=min_class - 1, random_state=42)
        else:
            sampler = RandomOverSampler(random_state=42)

        X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Define models
    models = {
        "classification": {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC(probability=True)
        },
        "regression": {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "SVR": SVR()
        }
    }

    best_model = None
    best_model_name = None
    best_score = -float("inf")
    best_report = None
    best_plot_path = None
    model_table = []

    for name, model in models[task_type].items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if is_classification:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            report = classification_report(y_test, y_pred, output_dict=True)
            plot_path = plot_conf_matrix(y_test, y_pred, name)
            score = f1
            model_table.append({
                "Model": name,
                "Accuracy": round(acc * 100, 2),
                "Macro F1": round(f1 * 100, 2),
                "Confusion Matrix": plot_path
            })
        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            report = {"R2 Score": round(r2, 4), "MSE": round(mse, 4)}
            plot_path = plot_regression(y_test, y_pred, name)
            score = r2
            model_table.append({
                "Model": name,
                "R² Score": round(r2, 4),
                "MSE": round(mse, 4),
                "Regression Plot": plot_path
            })

        if score > best_score:
            best_model = model
            best_model_name = name
            best_score = score
            best_report = report
            best_plot_path = plot_path

    # GridSearch on XGBoost
    print("🔍 Running GridSearch on XGBoost...")
    param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss') if is_classification else XGBRegressor()
    grid = GridSearchCV(xgb, param_grid, scoring='f1_macro' if is_classification else 'r2', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    xgb_best = grid.best_estimator_
    y_pred = xgb_best.predict(X_test)

    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        plot_path = plot_conf_matrix(y_test, y_pred, "XGBoost_Tuned")
        report = classification_report(y_test, y_pred, output_dict=True)
        score = f1
        model_table.append({
            "Model": "XGBoost (Tuned)",
            "Accuracy": round(acc * 100, 2),
            "Macro F1": round(f1 * 100, 2),
            "Confusion Matrix": plot_path
        })
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        plot_path = plot_regression(y_test, y_pred, "XGBoost_Tuned")
        report = {"R2 Score": round(r2, 4), "MSE": round(mse, 4)}
        score = r2
        model_table.append({
            "Model": "XGBoost (Tuned)",
            "R² Score": round(r2, 4),
            "MSE": round(mse, 4),
            "Regression Plot": plot_path
        })

    if score > best_score:
        best_model = xgb_best
        best_model_name = "XGBoost (Tuned)"
        best_score = score
        best_report = report
        best_plot_path = plot_path

    # Save best model
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model, {
        "Best Model": best_model_name,
        "Best Score": round(best_score * 100, 2) if is_classification else round(best_score, 4),
        "Evaluation Report": best_report,
        "Plot Path": best_plot_path,
        "Comparison Table": model_table,
        "Best Parameters": grid.best_params_,
        "Model File": "outputs/model.pkl",
        "Task Type": "classification" if is_classification else "regression"
    }