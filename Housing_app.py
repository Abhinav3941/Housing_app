# ===== CALIFORNIA HOUSING PRICE PREDICTION (FINAL WORKING VERSION) =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import gradio as gr

# ===== 1. DATA LOADING & FEATURE ENGINEERING =====
def load_and_engineer_data():
    """Load data and create enhanced features"""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['PRICE'] = data.target
    
    # Primary feature engineering
    df['RoomsPerHousehold'] = df['AveRooms'] / df['AveBedrms']
    df['BedroomsRatio'] = df['AveBedrms'] / df['AveRooms']
    
    # Advanced features
    df['IncomePerRoom'] = df['MedInc'] / df['AveRooms']
    df['CoastalProximity'] = np.abs(df['Latitude'] - 34) + np.abs(df['Longitude'] + 118)
    
    # Selected features based on importance analysis
    top_features = [
        'MedInc', 
        'Latitude', 
        'RoomsPerHousehold',
        'HouseAge',
        'Longitude',
        'IncomePerRoom',
        'CoastalProximity'
    ]
    
    return df, top_features

df, top_features = load_and_engineer_data()

# ===== 2. EXPLORATORY DATA ANALYSIS =====
def perform_eda(df):
    """Generate insights and visualizations"""
    print("\n=== DATA SUMMARY ===")
    print(df[top_features + ['PRICE']].describe())
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    # Plot distributions
    df[top_features].hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

perform_eda(df)

# ===== 3. MODEL TRAINING & OPTIMIZATION =====
def train_model(df, top_features):
    """Train and tune XGBoost model"""
    X = df[top_features]
    y = df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    model = XGBRegressor(random_state=42)
    grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Get best model
    best_model = grid.best_estimator_
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='r2')
    
    return best_model, X_test, y_test, cv_scores

best_model, X_test, y_test, cv_scores = train_model(df, top_features)

# ===== 4. EVALUATION =====
def evaluate_model(model, X_test, y_test, cv_scores):
    """Generate performance metrics and visuals"""
    y_pred = model.predict(X_test)
    
    print("\n=== MODEL PERFORMANCE ===")
    print(f"Best R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)) * 100000:,.2f}")
    print(f"Cross-validated R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    
    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X_test.columns)
    feature_importance.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Save predictions
    results = X_test.copy()
    results['Actual_Price'] = y_test
    results['Predicted_Price'] = y_pred
    results.to_csv('model_predictions.csv', index=False)

evaluate_model(best_model, X_test, y_test, cv_scores)

# ===== 5. DEPLOYMENT =====
def create_app(model, top_features):
    """Build interactive prediction interface"""
    def predict_price(MedInc, Latitude, Longitude, HouseAge, RoomsPerHousehold):
        try:
            # Convert and validate inputs
            MedInc = float(MedInc)
            Latitude = float(Latitude)
            Longitude = float(Longitude)
            HouseAge = float(HouseAge)
            RoomsPerHousehold = max(0.1, float(RoomsPerHousehold))  # Prevent division by zero
            
            # Calculate engineered features
            IncomePerRoom = MedInc / RoomsPerHousehold
            CoastalProximity = np.abs(Latitude - 34) + np.abs(Longitude + 118)
            
            # Create input array matching training format
            sample = pd.DataFrame([[MedInc, Latitude, RoomsPerHousehold, 
                                  HouseAge, Longitude, IncomePerRoom, 
                                  CoastalProximity]],
                                columns=top_features)
            
            # Predict and format
            prediction = model.predict(sample)[0] * 100000  # Convert to USD
            return f"Predicted Price: ${prediction:,.2f}"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=predict_price,
        inputs=[
            gr.Slider(0.5, 15, value=3.87, label="Median Income (in $100k)"),
            gr.Slider(32, 42, value=34, label="Latitude"),
            gr.Slider(-124, -114, value=-118, label="Longitude"),
            gr.Slider(1, 52, value=29, label="House Age"),
            gr.Slider(0.1, 15, value=5, label="Rooms per Household")  # Min=0.1
        ],
        outputs="text",
        title="California Housing Price Predictor",
        description="Predicts median home prices based on key features",
        examples=[
            [3.87, 34, -118, 29, 5],  # Average home
            [8.0, 37, -122, 15, 8]    # High-end coastal property
        ]
    )
    return interface

app = create_app(best_model, top_features)

# ===== 6. SAVE ARTIFACTS =====
joblib.dump(best_model, 'california_housing_model.pkl')
print("\nModel saved as 'california_housing_model.pkl'")

# Generate requirements file
with open('requirements.txt', 'w') as f:
    f.write("""numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.5.0
gradio>=3.0.0
joblib>=1.1.0""")

# ===== 7. LAUNCH APP =====
if __name__ == "__main__":
    print("\n=== SETUP COMPLETE ===")
    app.launch(server_port=7960, share=True)