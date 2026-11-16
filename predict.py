"""
Send a few California Housing rows to the served "champion" model and compare its
predictions with the actual values.

Start the model server first, in a separate terminal from this folder:
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
        -m "models:/CaliforniaHousingModel@champion" -p 5002 --env-manager local

Then run:
    python predict.py
"""
import requests
from sklearn.datasets import fetch_california_housing

URL = "http://127.0.0.1:5002/invocations"
TARGET = "MedHouseVal"


def main():
    data = fetch_california_housing(as_frame=True).frame
    sample = data.sample(5, random_state=7)
    features = sample.drop(columns=[TARGET])

    # "dataframe_split" is the request format MLflow's scoring server expects:
    # {"dataframe_split": {"columns": [...], "data": [[...], ...]}}
    payload = {"dataframe_split": features.to_dict(orient="split", index=False)}

    try:
        response = requests.post(URL, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {URL}. Start the model server first (see the top of this file).")
        return

    predictions = response.json()["predictions"]
    print(f"{'Median income':>14} {'House age':>10} {'Actual':>12} {'Predicted':>12}")
    for (_, row), pred in zip(sample.iterrows(), predictions):
        print(f"{row['MedInc']:>14.2f} {row['HouseAge']:>10.0f} "
              f"{row[TARGET] * 100_000:>12,.0f} {pred * 100_000:>12,.0f}")


if __name__ == "__main__":
    main()
