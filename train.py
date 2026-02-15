import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def train():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE:  {mean_absolute_error(y_test, preds):.4f}")
    print(f"R^2:  {r2_score(y_test, preds):.4f}")

    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")

    print(f"\nFeature names ({len(data.feature_names)}): {data.feature_names}")


if __name__ == "__main__":
    train()
