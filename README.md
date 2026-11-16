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

**3. Richer logging per run.** Each run logs its parameters, RMSE/MAE/R² metrics, the dataset (`mlflow.log_input`), tags (`model_family`, `dataset`), an input example, and a **predicted-vs-actual plot** as an artifact.

**4. SQLite tracking backend and Model Registry.** Tracking uses `sqlite:///mlflow.db` explicitly, which the original lab's file store didn't support for registration. The run with the lowest test RMSE is automatically registered as **`CaliforniaHousingModel`**, given the alias **`champion`**, and tagged in the runs table.

**5. Serving by registry alias instead of a hardcoded path.** The original notebooks load and serve models from paths like `mlruns/0/<run_id>/artifacts/model`, which only exist on the instructor's machine and don't match MLflow 3's storage layout. Here the server loads `models:/CaliforniaHousingModel@champion`, so it always serves the current best model. `predict.py` sends sample rows to the server using MLflow's `dataframe_split` request format and compares predictions with actual values.

The original lab files (`starter.ipynb`, `linear_regression.*`, `serving.*`, `course_notes.pdf`) are kept for reference.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Train and compare the models:**

```bash
python housing_experiment.py
```

**Open the MLflow UI** in a second terminal (activate the venv there too), then go to http://localhost:5001:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**Serve the champion model** in a third terminal:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m "models:/CaliforniaHousingModel@champion" -p 5002 --env-manager local
```

**Send predictions** from the first terminal:

```bash
python predict.py
```

Running `housing_experiment.py` again adds 7 new runs and registers a new model version, and the `champion` alias moves to the newest best model.
