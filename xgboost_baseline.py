import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
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
test_data = pd.read_csv('data/test_full.csv')

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
# Apply log(1+y) transformation to the target variable
y_train_log = np.log1p(y_train)

# Add interaction features (Duration × Heart_Rate)
X_train['Duration_Heart_Rate'] = X_train['Duration'] * X_train['Heart_Rate']
X_test['Duration_Heart_Rate'] = X_test['Duration'] * X_test['Heart_Rate']

# Update numerical features list
numerical_features.append('Duration_Heart_Rate')

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Get encoded feature names
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Combine numerical and encoded categorical features
X_train_processed = np.hstack([X_train[numerical_features].values, X_train_encoded])
X_test_processed = np.hstack([X_test[numerical_features].values, X_test_encoded])

# Create feature names for the processed data
feature_names = numerical_features + list(encoded_feature_names)

print("Training XGBoost model...")
# Create XGBoost regressor with recommended hyperparameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squaredlogerror',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model on the log-transformed target
xgb_model.fit(X_train_processed, y_train_log)

print("Evaluating model...")
# Perform 5-fold cross-validation with custom scorer
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train_processed, y_train_log, 
                           scoring=rmsle_scorer, 
                           cv=kf, n_jobs=-1)

# Convert to RMSLE scores (negative because we used negative RMSLE in scorer)
cv_rmsle_scores = -cv_scores
print(f"Cross-validation RMSLE scores: {cv_rmsle_scores}")
print(f"Mean CV RMSLE: {cv_rmsle_scores.mean():.4f} (+/- {cv_rmsle_scores.std() * 2:.4f})")

print("Generating predictions on test set...")
# Generate predictions on test set
test_predictions_log = xgb_model.predict(X_test_processed)

# Transform predictions back from log scale
test_predictions = np.expm1(test_predictions_log)

# Ensure no negative predictions (clip to 0)
test_predictions = np.clip(test_predictions, 0, None)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'Calories': test_predictions
})

print("Saving results...")
# Save model
joblib.dump({
    'model': xgb_model,
    'encoder': encoder,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'feature_names': feature_names
}, 'model_artifacts/xgboost_baseline_model.pkl')

# Save submission file
submission.to_csv('results/xgboost_baseline_submission.csv', index=False)

# Save feature importance
importance_scores = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_scores
}).sort_values(by='importance', ascending=False)

feature_importance.to_csv('results/xgboost_feature_importance.csv', index=False)

print("XGBoost baseline model implementation completed!")
print(f"Mean CV RMSLE: {cv_rmsle_scores.mean():.4f}")
print("Results saved to 'results/' directory")
print("Model saved to 'model_artifacts/' directory")