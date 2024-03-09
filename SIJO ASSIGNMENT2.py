# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Load the dataset (replace 'your_dataset.csv' with the actual file name)
data = pd.read_csv('your_dataset.csv')

# Explore the dataset
print(data.head())

# Preprocess the data
# For simplicity, let's assume 'features' contain all columns except the target variable, and 'target' is the column you want to predict.
features = data.drop('target', axis=1)
target = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
linear_predictions = linear_model.predict(X_test_scaled)

# Evaluate the model
mse_linear = mean_squared_error(y_test, linear_predictions)
print(f'Linear Regression Mean Squared Error: {mse_linear}')

# Train a Random Forest Regression model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
rf_predictions = random_forest_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest Mean Squared Error: {mse_rf}')

# Cross-validation on the entire dataset
linear_cv_scores = cross_val_score(linear_model, features, target, cv=5, scoring='neg_mean_squared_error')
rf_cv_scores = cross_val_score(random_forest_model, features, target, cv=5, scoring='neg_mean_squared_error')

print(f'Cross-validation Linear Regression Mean Squared Error: {abs(linear_cv_scores.mean())}')
print(f'Cross-validation Random Forest Mean Squared Error: {abs(rf_cv_scores.mean())}')
