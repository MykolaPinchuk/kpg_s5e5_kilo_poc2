import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define RMSLE function
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE)
    
    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values
    
    Returns:
    float: RMSLE score
    """
    # Ensure non-negative values for log calculation
    y_pred_clipped = np.clip(y_pred, 1e-15, None)
    y_true_clipped = np.clip(y_true, 1e-15, None)
    
    # Calculate RMSLE
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

# Custom scorer for cross-validation that handles negative predictions
def rmsle_scorer(estimator, X, y):
    """
    Custom scorer for cross-validation that ensures non-negative predictions
    """
    y_pred = estimator.predict(X)
    # Clip predictions to ensure non-negative values
    y_pred = np.clip(y_pred, 0, None)
    return -rmsle(y, y_pred)  # Negative because cross_val_score maximizes

# Load data
print("Loading data...")
train_data = pd.read_csv('data/train_subsample.csv')
test_data = pd.read_csv('data/test_subsample.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Define features and target
feature_columns = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
categorical_features = ['Sex']
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target_column = 'Calories'

# Separate features and target
X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_test = test_data[feature_columns]

print("Data preprocessing...")
# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create full pipeline with preprocessing and model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("Training model...")
# Train the model
model_pipeline.fit(X_train, y_train)

print("Evaluating model...")
# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Train on subset and evaluate
model_pipeline.fit(X_train_split, y_train_split)
y_val_pred = model_pipeline.predict(X_val_split)

# Clip predictions to ensure non-negative values
y_val_pred = np.clip(y_val_pred, 0, None)

# Calculate RMSLE on validation set
val_rmsle = rmsle(y_val_split, y_val_pred)
print(f"Validation RMSLE: {val_rmsle:.4f}")

print("Performing cross-validation...")
# Perform 5-fold cross-validation with custom scorer
cv_scores = cross_val_score(model_pipeline, X_train, y_train, 
                           scoring=rmsle_scorer, 
                           cv=5, n_jobs=-1)

# Convert to RMSLE scores (negative because we used negative RMSLE in scorer)
cv_rmsle_scores = -cv_scores
print(f"Cross-validation RMSLE scores: {cv_rmsle_scores}")
print(f"Mean CV RMSLE: {cv_rmsle_scores.mean():.4f} (+/- {cv_rmsle_scores.std() * 2:.4f})")

print("Generating predictions on test set...")
# Generate predictions on test set
test_predictions = model_pipeline.predict(X_test)

# Ensure no negative predictions (clip to 0)
test_predictions = np.clip(test_predictions, 0, None)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'Calories': test_predictions
})

print("Saving results...")
# Save model pipeline
joblib.dump(model_pipeline, 'model_artifacts/baseline_model_pipeline.pkl')

# Save submission file
submission.to_csv('results/baseline_submission.csv', index=False)

# Save feature importance
feature_names = (numerical_features + 
                list(model_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

coefficients = model_pipeline.named_steps['regressor'].coef_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values(by='coefficient', key=abs, ascending=False)

feature_importance.to_csv('results/feature_importance.csv', index=False)

print("Baseline model implementation completed!")
print(f"Validation RMSLE: {val_rmsle:.4f}")
print(f"Mean CV RMSLE: {cv_rmsle_scores.mean():.4f}")
print("Results saved to 'results/' directory")
print("Model saved to 'model_artifacts/' directory")