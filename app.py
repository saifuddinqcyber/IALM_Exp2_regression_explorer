from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

def build_datasets():
    # Small, reproducible teaching datasets; all data is available locally.
    study_hours = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5], dtype=float)
    exam_score = np.array([34,39,41,47,45,53,58,55,63,64,70,68,76,78,81,85,87,91,93,96], dtype=float)

    ad_spend = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5], dtype=float)
    sales = np.array([13,16,18,20,22,23,27,29,31,30,35,37,39,41,44,46,47,51,52,55], dtype=float)

    temperature = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5], dtype=float)
    energy = np.array([91,81,76,66,59,53,45,40,35,31,29,30,32,38,45,54,64,77,91,108], dtype=float)

    diabetes = load_diabetes()
    # Use one meaningful predictor so all datasets can be visualized as a 2D regression fit.
    bmi_index = list(diabetes.feature_names).index("bmi")
    diabetes_x = diabetes.data[:, bmi_index] * 10 + 30
    diabetes_y = diabetes.target

    return {
        "study": {"name": "Study hours vs exam score", "x_name": "Study hours", "y_name": "Exam score", "X": study_hours.reshape(-1,1), "y": exam_score, "description": "Illustrative student study-hours dataset."},
        "advertising": {"name": "Advertising spend vs sales", "x_name": "Advertising spend (units)", "y_name": "Sales (units)", "X": ad_spend.reshape(-1,1), "y": sales, "description": "Illustrative advertising and sales dataset."},
        "energy": {"name": "Temperature vs energy use", "x_name": "Temperature index", "y_name": "Energy use", "X": temperature.reshape(-1,1), "y": energy, "description": "Illustrative curved relationship between temperature and energy use."},
        "diabetes": {"name": "Diabetes dataset: BMI vs disease progression", "x_name": "BMI (scaled)", "y_name": "Disease progression", "X": diabetes_x.reshape(-1,1), "y": diabetes_y, "description": "Scikit-learn diabetes dataset, using BMI as a single predictor for clear 2D visualization."},
    }

DATASETS = build_datasets()

def model_factories():
    return {
        "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial Regression (degree 2)": lambda: make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=0.5)),
        "Random Forest Regression": lambda: RandomForestRegressor(n_estimators=160, max_depth=5, min_samples_leaf=2, random_state=42),
    }

def evaluate_dataset(dataset_key, test_size=0.2, regression_method="Best performing model", input_feature=None):
    data = DATASETS[dataset_key]
    X, y = data["X"], data["y"]
    # Built-in teaching datasets currently expose one input feature each.
    if input_feature and input_feature != data["x_name"]:
        raise ValueError("The selected input feature is not available for this dataset.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42
    )
    rows = []
    fitted = {}
    for name, factory in model_factories().items():
        model = factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rows.append({
            "model": name,
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)),
        })
        fitted[name] = model

    # Best model: primarily lowest RMSE; tie-break using higher R².
    rows.sort(key=lambda row: (row["rmse"], -row["r2"]))
    available_models = {row["model"] for row in rows}
    if regression_method == "Best performing model":
        best_name = rows[0]["model"]
    elif regression_method in available_models:
        best_name = regression_method
    else:
        raise ValueError("Unknown regression method selected.")
    best_model = fitted[best_name]
    x_min, x_max = float(np.min(X)), float(np.max(X))
    curve_x = np.linspace(x_min, x_max, 120).reshape(-1, 1)
    curve_y = best_model.predict(curve_x)

    # Keep actual test rows in original order for readable plots.
    best_test_predictions = best_model.predict(X_test)
    residuals = y_test - best_test_predictions
    return {
        "dataset": {"key": dataset_key, "name": data["name"], "x_name": data["x_name"], "y_name": data["y_name"], "description": data["description"], "observations": int(len(y))},
        "metrics": rows,
        "best_model": best_name,
        "selected_method": regression_method,
        "input_feature": data["x_name"],
        "fit": {
            "x": X.flatten().astype(float).tolist(),
            "y": y.astype(float).tolist(),
            "curve_x": curve_x.flatten().astype(float).tolist(),
            "curve_y": curve_y.astype(float).tolist(),
        },
        "test": {
            "actual": y_test.astype(float).tolist(),
            "predicted": best_test_predictions.astype(float).tolist(),
            "residuals": residuals.astype(float).tolist(),
        },
        "test_size": float(test_size),
    }

@app.route("/")
def index():
    dataset_options = [{"key": key, "name": value["name"]} for key, value in DATASETS.items()]
    return render_template("index.html", datasets=dataset_options)

@app.route("/api/dataset/<dataset_key>")
def dataset_preview(dataset_key):
    if dataset_key not in DATASETS:
        return jsonify({"error": "Unknown dataset selected."}), 404
    data = DATASETS[dataset_key]
    rows = [{"x": float(x), "y": float(y)} for x, y in zip(data["X"].flatten()[:10], data["y"][:10])]
    return jsonify({
        "name": data["name"],
        "x_name": data["x_name"],
        "y_name": data["y_name"],
        "observations": int(len(data["y"])),
        "rows": rows,
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    dataset_key = payload.get("dataset", "study")
    if dataset_key not in DATASETS:
        return jsonify({"error": "Unknown dataset selected."}), 400
    try:
        test_size = float(payload.get("test_size", 0.2))
    except (TypeError, ValueError):
        return jsonify({"error": "Test split must be a number."}), 400
    if not 0.1 <= test_size <= 0.4:
        return jsonify({"error": "Choose a test split between 10% and 40%."}), 400
    try:
        regression_method = payload.get("regression_method", "Best performing model")
        input_feature = payload.get("input_feature")
        return jsonify(evaluate_dataset(dataset_key, test_size, regression_method, input_feature))
    except Exception as exc:
        app.logger.exception("Regression analysis failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
