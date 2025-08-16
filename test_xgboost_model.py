import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessing components
model_artifacts = joblib.load('model_artifacts/xgboost_baseline_model.pkl')
xgb_model = model_artifacts['model']
encoder = model_artifacts['encoder']
numerical_features = model_artifacts['numerical_features']
categorical_features = model_artifacts['categorical_features']

print("Model and preprocessing components loaded successfully!")

# Create a sample test instance
sample_data = pd.DataFrame({
    'Sex': ['male'],
    'Age': [30],
    'Height': [180.0],
    'Weight': [75.0],
    'Duration': [30.0],
    'Heart_Rate': [100.0],
    'Body_Temp': [40.0]
})

# Add interaction feature
sample_data['Duration_Heart_Rate'] = sample_data['Duration'] * sample_data['Heart_Rate']

# Apply the same preprocessing
sample_encoded = encoder.transform(sample_data[categorical_features])
sample_processed = np.hstack([sample_data[numerical_features].values, sample_encoded])

# Make a prediction
prediction_log = xgb_model.predict(sample_processed)
prediction = np.expm1(prediction_log)

print(f"Sample prediction: {prediction[0]:.2f} calories")
print("Model test completed successfully!")