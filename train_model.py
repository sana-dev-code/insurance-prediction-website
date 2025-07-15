import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load dataset (adjust path if needed)
df = pd.read_csv('insurance.csv')  # your CSV file

# Check columns
print(df.columns)  # ensure target column name

# Prepare features and target
X = df.drop('expenses', axis=1)  # 'expenses' is target column
y = df['expenses']

# Categorical columns to encode
categorical_cols = ['sex', 'smoker', 'region']

# Preprocessing pipeline for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Create full pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train
model.fit(X_train, y_train)

# Save model with pickle
with open('insurance_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as insurance_model.pkl")
