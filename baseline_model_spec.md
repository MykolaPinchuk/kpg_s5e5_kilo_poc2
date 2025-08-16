# Baseline Model Architecture Specification

## Overview
This document outlines the technical specification for a simple Linear Regression baseline model to predict calories burned during workouts. This is intended as a straightforward proof-of-concept (POC) for the regression problem.

## Data Preprocessing

### Categorical Features
- **Gender**: One-hot encoding will be used to convert the categorical gender feature into numerical format
  - Creates two binary features: `Gender_Male` and `Gender_Female`

### Numerical Features
- **Age, Height, Weight, Duration, Heart_Rate, Body_Temp**: Standard scaling (z-score normalization) will be applied
  - Formula: `(x - mean) / std_dev`
  - This ensures all numerical features contribute equally to the model

### Data Splitting
- Training/validation split for initial model evaluation (80/20 split)
- Cross-validation (5-fold) for robust performance estimation

## Model Selection

### Chosen Model: Linear Regression
**Rationale:**
1. **Simplicity**: Straightforward to implement and understand
2. **Interpretability**: Model coefficients provide insights into feature importance
3. **Performance benchmark**: Serves as a reference point for more complex models
4. **Computational efficiency**: Fast training and prediction times
5. **Suitability for regression**: Directly addresses the continuous target variable (Calories)

## Evaluation Approach

### Primary Metric: Root Mean Squared Logarithmic Error (RMSLE)
- Formula: `sqrt(mean((log(1 + y_pred) - log(1 + y_true))^2))`
- Penalizes underprediction more heavily than overprediction
- Particularly suitable when the target variable has a wide range of values

### Validation Strategy
- 5-fold Cross-Validation for robust performance estimation
- Train/validation split for initial hyperparameter tuning

## Expected Outputs

### Model Artifacts
- Trained Linear Regression model
- Preprocessing pipeline (one-hot encoding + scaling)
- Feature importance analysis (coefficients)

### Performance Metrics
- RMSLE score on validation set
- Cross-validation scores with mean and standard deviation

### Submission Files
- Predictions on test set in the required submission format

## Model Interface

### Methods
- `fit(X, y)`: Train the model on features and target
- `predict(X)`: Generate predictions for new data
- `score(X, y)`: Calculate RMSLE score
- `get_feature_importance()`: Return feature coefficients

## Implementation Plan

### Dependencies
- pandas for data manipulation
- scikit-learn for preprocessing and modeling
- numpy for numerical operations

### Pipeline Steps
1. Load and explore data
2. Apply one-hot encoding to Gender feature
3. Apply standard scaling to numerical features
4. Train Linear Regression model
5. Evaluate using RMSLE metric
6. Generate predictions for test set
7. Create submission file

## Next Steps
1. Implement the baseline model according to this specification
2. Evaluate performance and document results
3. Use this as a benchmark for more complex models