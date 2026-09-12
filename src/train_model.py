import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("data/salary_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==============================
# 2. Remove Unnecessary Columns
# ==============================

df = df.drop(
    columns=["salary", "salary_currency"],
    errors="ignore"
)

df = df.dropna(subset=["salary_in_usd"])


# ==============================
# 3. Features & Target
# ==============================

X = df.drop(columns=["salary_in_usd"])
y = df["salary_in_usd"]


# ==============================
# 4. Identify Column Types
# ==============================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ==============================
# 5. Preprocessing
# ==============================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==============================
# 6. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 7. Models
# ==============================

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

gradient_boosting = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

extra_trees = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

voting_model = VotingRegressor([
    ("random_forest", random_forest),
    ("gradient_boosting", gradient_boosting),
    ("extra_trees", extra_trees)
])

    
# ==============================
# 8. Create Pipelines
# ==============================

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", random_forest)
])

gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gradient_boosting)
])

voting_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", voting_model)
])

# ==============================
# 9. Train Random Forest
# ==============================

print("\nTraining Random Forest...")

rf_pipeline.fit(X_train, y_train)

print("Random Forest training completed!")


#==============================
# 9.1 Train Voting Regressor
#==============================
print("\nTraining Voting Regressor...")

voting_pipeline.fit(X_train, y_train)

print("Voting Regressor training completed!")


voting_pred = voting_pipeline.predict(X_test)

voting_mae = mean_absolute_error(y_test, voting_pred)
voting_mse = mean_squared_error(y_test, voting_pred)
voting_rmse = voting_mse ** 0.5
voting_r2 = r2_score(y_test, voting_pred)

print("\nVoting Regressor:")
print(f"MAE  : ${voting_mae:,.2f}")
print(f"RMSE : ${voting_rmse:,.2f}")
print(f"R²   : {voting_r2:.4f}")

# ==============================
# 10. Random Forest Prediction
# ==============================

rf_pred = rf_pipeline.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = rf_mse ** 0.5
rf_r2 = r2_score(y_test, rf_pred)


# ==============================
# 11. Train Gradient Boosting
# ==============================

print("\nTraining Gradient Boosting...")

gb_pipeline.fit(X_train, y_train)

print("Gradient Boosting training completed!")


# ==============================
# 12. Gradient Boosting Prediction
# ==============================

gb_pred = gb_pipeline.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_rmse = gb_mse ** 0.5
gb_r2 = r2_score(y_test, gb_pred)


# ==============================
# 13. Compare Models
# ==============================

print("\n============================================")
print("           MODEL COMPARISON")
print("============================================")

print("\nRandom Forest:")
print(f"MAE  : ${rf_mae:,.2f}")
print(f"RMSE : ${rf_rmse:,.2f}")
print(f"R²   : {rf_r2:.4f}")

print("\nGradient Boosting:")
print(f"MAE  : ${gb_mae:,.2f}")
print(f"RMSE : ${gb_rmse:,.2f}")
print(f"R²   : {gb_r2:.4f}")

print("\n============================================")


print("\n============================================")
print("             FINAL COMPARISON")
print("============================================")

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Voting Regressor"
    ],
    "MAE": [
        rf_mae,
        gb_mae,
        voting_mae
    ],
    "RMSE": [
        rf_rmse,
        gb_rmse,
        voting_rmse
    ],
    "R2": [
        rf_r2,
        gb_r2,
        voting_r2
    ]
})
# Save model comparison results
results.to_csv(
    "src/model_results.csv",
    index=False
)

print("\nModel results saved to src/model_results.csv")

print(results.to_string(index=False))

print("\nBest Model:")
print(
    results.loc[
        results["R2"].idxmax(),
        "Model"
    ]
)

# ==============================
# Select Best Model
# ==============================

model_pipelines = {
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline,
    "Voting Regressor": voting_pipeline
}

best_model_name = results.loc[
    results["R2"].idxmax(),
    "Model"
]

best_model = model_pipelines[best_model_name]

print("\n============================================")
print("BEST MODEL")
print("============================================")
print(f"Selected Model: {best_model_name}")
print(f"R² Score: {results.loc[results['R2'].idxmax(), 'R2']:.4f}")


# ==============================
# Save Best Model
# ==============================

joblib.dump(
    best_model,
    "models/salary_model.pkl"
)

print("\nBest model saved successfully!")
print("Location: models/salary_model.pkl")