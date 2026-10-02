import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv("logistics_delivery_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 2. FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]


# ---------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ---------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------

numeric_features = [
    "distance_km",
    "traffic_index",
    "package_weight_kg",
    "driver_experience_yr",
    "weather_severity",
    "is_peak_hour",
    "num_stops_enroute"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)


# ---------------------------------------------------------
# 5. DEFINE MODELS
# ---------------------------------------------------------

models = {

    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ---------------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# ---------------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    trained_models[name] = pipeline

    print("\n--------------------------------")
    print(name)
    print("--------------------------------")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.4f}")


# ---------------------------------------------------------
# 7. MODEL COMPARISON TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


# ---------------------------------------------------------
# 8. SELECT BEST MODEL
# ---------------------------------------------------------

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nBest model based on RMSE:")
print(best_model_name)


# ---------------------------------------------------------
# 9. CROSS-VALIDATION
# ---------------------------------------------------------

print("\nCROSS-VALIDATION RESULTS")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=kf,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = np.sqrt(-scores)

    print(
        f"{name}: "
        f"Mean CV RMSE = {cv_rmse.mean():.3f}"
    )


# ---------------------------------------------------------
# 10. ACTUAL VS PREDICTED
# ---------------------------------------------------------

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title(
    f"Actual vs Predicted Delivery Time - {best_model_name}"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 11. MODEL PERFORMANCE CHART
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.ylabel("RMSE (minutes)")
plt.xlabel("Model")
plt.title("Model Performance Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "model_performance_comparison.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 12. FEATURE IMPORTANCE
# ---------------------------------------------------------

if best_model_name in [
    "Random Forest",
    "Gradient Boosting"
]:

    model_object = best_model.named_steps["model"]

    importances = model_object.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": numeric_features,
        "Importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        "Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print(feature_importance_df)

    feature_importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Feature Importance - {best_model_name}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 13. DISTANCE VS DELIVERY TIME
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["distance_km"],
    df["delivery_time_min"],
    alpha=0.4
)

plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.title("Distance vs Delivery Time")

plt.tight_layout()

plt.savefig(
    "distance_vs_delivery_time.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 14. TRAFFIC VS DELIVERY TIME
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["traffic_index"],
    df["delivery_time_min"],
    alpha=0.4
)

plt.xlabel("Traffic Index")
plt.ylabel("Delivery Time (minutes)")
plt.title("Traffic Index vs Delivery Time")

plt.tight_layout()

plt.savefig(
    "traffic_vs_delivery_time.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 15. FINAL SUMMARY
# ---------------------------------------------------------

print("\n========================================")
print("PREDICTIVE MODELING COMPLETED")
print("========================================")

print(f"Best Model: {best_model_name}")

best_row = results_df[
    results_df["Model"] == best_model_name
].iloc[0]

print(f"RMSE: {best_row['RMSE']:.3f} minutes")
print(f"MAE : {best_row['MAE']:.3f} minutes")
print(f"R²  : {best_row['R2']:.4f}")

print("\nFiles generated:")
print("- model_comparison_results.csv")
print("- actual_vs_predicted.png")
print("- model_performance_comparison.png")
print("- distance_vs_delivery_time.png")
print("- traffic_vs_delivery_time.png")

if best_model_name in [
    "Random Forest",
    "Gradient Boosting"
]:
    print("- feature_importance.csv")
    print("- feature_importance.png")

print("\nWeek 4 predictive modeling task completed!")