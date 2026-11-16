# MLflow Lab 1: Experiment Tracking on California Housing

This repo is my version of [MLflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1) from the Northeastern MLOps course. The original lab tracks a single ElasticNet model on the red wine quality dataset and serves a model from a hardcoded run path. My version turns it into a full model comparison on a different dataset, with the best model registered and served by name.

## My modifications

**1. Different dataset: California Housing.** Instead of predicting wine quality, the models predict the median house value of a California district from 8 features such as median income, house age, and location (20,640 rows, from scikit-learn).

**2. Several models compared in one experiment.** `housing_experiment.py` trains 7 runs in the `california-housing-regression` experiment, each with a descriptive run name:

| Model | Variants |
|---|---|
| Linear Regression (baseline) | 1 |
| ElasticNet | `alpha` = 0.01, 0.1 |
| Random Forest | `n_estimators` = 50, 100 |
| Histogram Gradient Boosting | `learning_rate` = 0.05, 0.1 |

Linear models are wrapped in a pipeline with `StandardScaler`, since the features are on very different scales.

**3. Richer logging per run.** Each run logs its parameters, RMSE/MAE/R² metrics, the dataset (`mlflow.log_input`), tags (`model_family`, `dataset`), an input example, and a predicted-vs-actual plot as an artifact.

**4. Safe model serialization for tree models.** MLflow 3 saves scikit-learn models with skops, a safer format than pickle, which won't load tree models' internal node storage unless it's explicitly trusted. The script checks each model's untrusted types, trusts only scikit-learn's own internal types (the models were trained by the script itself), rejects anything else, and records the trusted types as a run tag.

**5. SQLite tracking backend and Model Registry.** Tracking uses `sqlite:///mlflow.db` explicitly, which enables the Model Registry (the original lab's file store skipped registration). The run with the lowest test RMSE is automatically registered as **`CaliforniaHousingModel`**, given the alias **`champion`**, and tagged in the runs table.

**6. Serving by registry alias instead of a hardcoded path.** The original notebooks load and serve models from paths like `mlruns/0/<run_id>/artifacts/model`, which only exist on the instructor's machine and don't match MLflow 3's storage layout. Here the server loads `models:/CaliforniaHousingModel@champion`, so it always serves the current best model. `predict.py` sends sample rows to the server using MLflow's `dataframe_split` request format and compares predictions with actual values.

The original lab files (`starter.ipynb`, `linear_regression.*`, `serving.*`, `course_notes.pdf`) are kept for reference.

## Results

Test-set results from my run (15,480 training rows, 5,160 test rows). The target is in units of $100,000, so an RMSE of 0.448 means a typical error of about $45,000.

| Run | RMSE | MAE | R² |
|---|---|---|---|
| **HistGradientBoosting lr=0.1 (champion)** | **0.448** | **0.297** | **0.848** |
| HistGradientBoosting lr=0.05 | 0.456 | 0.304 | 0.843 |
| Random Forest n_estimators=100 | 0.520 | 0.346 | 0.795 |
| Random Forest n_estimators=50 | 0.523 | 0.347 | 0.793 |
| ElasticNet alpha=0.01 | 0.733 | 0.531 | 0.594 |
| Linear Regression | 0.736 | 0.530 | 0.591 |
| ElasticNet alpha=0.1 | 0.792 | 0.594 | 0.526 |

What the comparison shows:
- **Tree ensembles clearly beat the linear models.** The champion's RMSE is 39% lower than the Linear Regression baseline (0.448 vs. 0.736). House prices depend on location and income in non-linear ways that trees capture and straight lines can't.
- **Stronger regularization hurt ElasticNet.** Raising `alpha` from 0.01 to 0.1 increased RMSE from 0.733 to 0.792.
- **More trees barely helped the Random Forest.** Going from 50 to 100 trees improved RMSE only from 0.523 to 0.520.

### Predictions from the served model

Output of `predict.py`, sending 5 districts to the champion model served at `http://127.0.0.1:5002/invocations`. Median income is in units of $10,000, and the dataset caps house values at $500,001.

| Median income | House age | Actual value | Predicted value |
|---|---|---|---|
| 2.03 | 31 | $360,000 | $315,659 |
| 4.31 | 30 | $336,000 | $303,711 |
| 3.67 | 52 | $269,900 | $263,413 |
| 2.64 | 26 | $287,500 | $313,570 |
| 9.20 | 23 | $500,001 | $488,807 |

Tested with Python 3.15 and MLflow 3.18.

## How to run

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

**Train and compare the models:**

~~~bash
python housing_experiment.py
~~~

**Open the MLflow UI** in a second terminal (activate the venv there too), then go to http://localhost:5001:

~~~bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
~~~

**Serve the champion model** in a third terminal:

~~~bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m "models:/CaliforniaHousingModel@champion" -p 5002 --env-manager local
~~~

**Send predictions** from the first terminal:

~~~bash
python predict.py
~~~

Running `housing_experiment.py` again adds 7 new runs and registers a new model version, and the `champion` alias moves to the newest best model.