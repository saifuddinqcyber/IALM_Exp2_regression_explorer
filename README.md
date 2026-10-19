# Regression Analysis & Model Comparison

A Flask web application that trains and compares regression models across multiple datasets and visualizes the fit, predictions, residuals, and evaluation metrics.

## Features

- Four datasets: study hours/exam score, advertising spend/sales, temperature/energy use, and the scikit-learn diabetes dataset (BMI as one predictor).
- Three regression techniques: Linear Regression, Polynomial Regression (degree 2 with Ridge regularization), and Random Forest Regression.
- MAE, MSE, RMSE, and R² on a held-out test set.
- Regression fit, actual vs. predicted, residual plot, metric comparison table, and normalized metric bar charts.
- Adjustable train/test split.

## 1. Install Python

Install Python 3.10 or newer from https://www.python.org/downloads/. On Windows, enable **Add Python to PATH** during setup.

## 2. Open the project folder

Extract the ZIP, then open a terminal in the `regression_analysis_flask` folder.

## 3. Create a virtual environment

### Windows (Command Prompt)

```bat
py -m venv .venv
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the Flask app

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Project structure

```text
regression_analysis_flask/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

## How it works

1. Select a dataset and test split.
2. Flask separates the dataset into training and test sets using a fixed random seed.
3. Each regression model is fitted on the training set.
4. The app calculates MAE, MSE, RMSE, and R² using only the held-out test set.
5. The model with the lowest RMSE is highlighted, with R² used as a tie-breaker.
6. Plotly renders the fit, actual-vs-predicted, residual, and metric-comparison visualizations.

## Metric interpretation

- **MAE**: average absolute prediction error; lower is better.
- **MSE**: average squared prediction error; lower is better.
- **RMSE**: square root of MSE, in the target variable's units; lower is better.
- **R²**: relative fit against predicting the test-set mean; higher is generally better. It can be negative on test data.

## Notes

- The study, advertising, and energy examples are illustrative datasets.
- The diabetes dataset comes from scikit-learn and is available locally through the installed package. This app uses BMI as a single predictor so its regression fit can be plotted in two dimensions; this is not a full multivariable diabetes model.
- Plotly is loaded from its public CDN, so chart rendering requires internet access. The Flask app and model training run locally.
- `debug=True` is for local development only. Do not use the development server as a production deployment.
