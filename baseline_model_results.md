# Baseline Linear Regression Model Results

## Model Overview
A baseline Linear Regression model was implemented to predict calories burned during workouts based on the provided specification.

## Data Preprocessing
- **Categorical Features**: One-hot encoding applied to the 'Sex' feature, creating binary features for male/female
- **Numerical Features**: Standard scaling (z-score normalization) applied to Age, Height, Weight, Duration, Heart_Rate, and Body_Temp
- **Data Split**: 80/20 train/validation split for initial evaluation

## Model Performance
- **Validation RMSLE**: 0.5566
- **Cross-Validation RMSLE (5-fold)**: 0.5653 (+/- 0.0227)

## Feature Importance (Coefficients)
| Feature | Coefficient |
|---------|-------------|
| Duration | 56.47 |
| Heart_Rate | 18.38 |
| Body_Temp | -14.22 |
| Age | 8.07 |
| Weight | 3.60 |
| Height | -1.82 |
| Sex_male | -1.45 |

## Key Observations
1. **Duration** has the highest positive impact on calories burned
2. **Heart_Rate** also significantly contributes to higher calorie burn
3. **Body_Temp** has a negative relationship with calories burned
4. The model successfully handles the regression task with reasonable accuracy

## Files Generated
- `results/baseline_submission.csv`: Predictions on test set
- `results/feature_importance.csv`: Feature coefficients
- `model_artifacts/baseline_model_pipeline.pkl`: Trained model pipeline

## Next Steps
This baseline model serves as a reference point for more complex models. The RMSLE score provides a benchmark for evaluating future improvements.