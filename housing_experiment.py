"""
California Housing price prediction: compare regression models with MLflow.

This is my modified version of the lab's linear_regression.py:
- Dataset: California Housing (20,640 districts, 8 features) instead of red wine quality.
- Models: Linear Regression, ElasticNet, Random Forest, and Histogram Gradient Boosting,
  each logged as its own named run inside one experiment, so they can be compared in the UI.
- Tracking: an explicit SQLite backend (sqlite:///mlflow.db), which also enables the Model Registry.
- Extra logging per run: the dataset, tags, an input example, and a predicted-vs-actual plot.
- The run with the lowest test RMSE is registered as "CaliforniaHousingModel" with the alias
  "champion", so it can be loaded or served by name instead of by a hardcoded run path.

Run it from this folder:
    python housing_experiment.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")  # draw plots in memory without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "california-housing-regression"
REGISTERED_MODEL_NAME = "CaliforniaHousingModel"
TARGET = "MedHouseVal"  # median house value, in units of $100,000
RANDOM_STATE = 42


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def candidate_models():
    """Return (run name, model family, hyperparameters to log, untrained model) for each run.

    Linear models are wrapped in a pipeline with StandardScaler because the features are on very
    different scales (e.g. population vs. median income). Tree models don't need scaling.
    """
    candidates = [
        ("Linear Regression", "linear", {},
         make_pipeline(StandardScaler(), LinearRegression())),
    ]
    for alpha in [0.01, 0.1]:
        candidates.append((
            f"ElasticNet alpha={alpha}", "linear", {"alpha": alpha, "l1_ratio": 0.5},
            make_pipeline(StandardScaler(),
                          ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=RANDOM_STATE)),
        ))
    for n_estimators in [50, 100]:
        candidates.append((
            f"Random Forest n_estimators={n_estimators}", "tree_ensemble",
            {"n_estimators": n_estimators, "max_depth": 12},
            RandomForestRegressor(n_estimators=n_estimators, max_depth=12, n_jobs=-1,
                                  random_state=RANDOM_STATE),
        ))
    for learning_rate in [0.05, 0.1]:
        candidates.append((
            f"HistGradientBoosting lr={learning_rate}", "tree_ensemble",
            {"learning_rate": learning_rate, "max_iter": 300},
            HistGradientBoostingRegressor(learning_rate=learning_rate, max_iter=300,
                                          random_state=RANDOM_STATE),
        ))
    return candidates


def trusted_sklearn_types(model):
    """Return the types skops won't load by default for this model, after checking they're all
    scikit-learn's own internals.

    MLflow saves scikit-learn models with skops, which refuses to load some internal types
    (for example the node storage inside Random Forest trees) unless they're explicitly trusted.
    These models were trained by this script, so trusting scikit-learn's own types is safe.
    Anything from another package is rejected instead of trusted automatically.
    """
    untrusted = sio.get_untrusted_types(data=sio.dumps(model))
    not_sklearn = [t for t in untrusted if not t.startswith("sklearn.")]
    if not_sklearn:
        raise ValueError(f"Refusing to trust types outside scikit-learn: {not_sklearn}")
    return untrusted


def pred_vs_actual_plot(actual, pred, title):
    """Scatter plot of predicted vs. actual values; points on the red line are perfect predictions."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, pred, s=4, alpha=0.3)
    limits = [0, 5.5]
    ax.plot(limits, limits, "r--", linewidth=1)
    ax.set(xlim=limits, ylim=limits, title=title,
           xlabel="Actual value ($100k)", ylabel="Predicted value ($100k)")
    fig.tight_layout()
    return fig


def main():
    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load the data (downloaded once, then cached by scikit-learn) and split 75/25.
    data = fetch_california_housing(as_frame=True).frame
    train, test = train_test_split(data, test_size=0.25, random_state=RANDOM_STATE)
    train_x, train_y = train.drop(columns=[TARGET]), train[TARGET]
    test_x, test_y = test.drop(columns=[TARGET]), test[TARGET]
    dataset = mlflow.data.from_pandas(data, name="california_housing", targets=TARGET)

    print(f"Training {len(candidate_models())} models on {len(train_x):,} rows "
          f"(testing on {len(test_x):,})\n")

    results = []
    for run_name, family, params, model in candidate_models():
        with mlflow.start_run(run_name=run_name) as run:
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, pred)

            estimator = model[-1] if hasattr(model, "steps") else model
            mlflow.log_input(dataset, context="training")
            mlflow.set_tags({"model_family": family, "dataset": "california_housing"})
            mlflow.log_param("model_type", type(estimator).__name__)
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            fig = pred_vs_actual_plot(test_y, pred, run_name)
            mlflow.log_figure(fig, "pred_vs_actual.png")
            plt.close(fig)

            signature = infer_signature(test_x, pred)
            trusted_types = trusted_sklearn_types(model)
            if trusted_types:
                mlflow.set_tag("skops_trusted_types", ", ".join(trusted_types))
            model_info = mlflow.sklearn.log_model(
                model, name="model", signature=signature, input_example=train_x.head(3),
                skops_trusted_types=trusted_types or None,
            )

            results.append({"run_name": run_name, "run_id": run.info.run_id,
                            "model_uri": model_info.model_uri,
                            "rmse": rmse, "mae": mae, "r2": r2})
            print(f"  {run_name:<34} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")

    # Pick the run with the lowest test RMSE and register it in the Model Registry.
    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    best = results_df.iloc[0]

    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    client.update_model_version(
        REGISTERED_MODEL_NAME, version.version,
        description=f"{best['run_name']} (test RMSE {best['rmse']:.3f}, R2 {best['r2']:.3f})",
    )
    client.set_tag(best["run_id"], "champion", "true")

    print("\nRanking by test RMSE (lower is better):")
    print(results_df[["run_name", "rmse", "mae", "r2"]].round(3).to_string(index=False))
    print(f"\nBest model: {best['run_name']}")
    print(f"Registered as {REGISTERED_MODEL_NAME} version {version.version}, alias 'champion'")
    print("\nView the runs:   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print("Serve the model: MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve "
          f"-m \"models:/{REGISTERED_MODEL_NAME}@champion\" -p 5002 --env-manager local")


if __name__ == "__main__":
    main()
