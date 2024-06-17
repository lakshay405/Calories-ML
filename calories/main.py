import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load the calorie data from CSV file into a DataFrame
calories_df = pd.read_csv('/content/calories.csv')
# Display the first few rows of the DataFrame
print(calories_df.head())

# Load the exercise data from CSV file into a DataFrame
exercise_df = pd.read_csv('/content/exercise.csv')
# Display the first few rows of the DataFrame
print(exercise_df.head())

# Combine the exercise data with calorie information
merged_data = pd.concat([exercise_df, calories_df['Calories']], axis=1)
# Display the first few rows of the combined DataFrame
print(merged_data.head())

# Check the dimensions of the DataFrame
print(merged_data.shape)

# Get detailed information about the DataFrame
print(merged_data.info())

# Check for missing values in the DataFrame
print(merged_data.isnull().sum())

# Get statistical summary of the DataFrame
print(merged_data.describe())

# Set up the seaborn style for the plots
sns.set()

# Plot the distribution of the 'Gender' column
plt.figure(figsize=(6, 6))
sns.countplot(x='Gender', data=merged_data)
plt.title('Gender Distribution')
plt.show()

# Plot the distribution of the 'Age' column
plt.figure(figsize=(6, 6))
sns.histplot(merged_data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Plot the distribution of the 'Height' column
plt.figure(figsize=(6, 6))
sns.histplot(merged_data['Height'], kde=True)
plt.title('Height Distribution')
plt.show()

# Plot the distribution of the 'Weight' column
plt.figure(figsize=(6, 6))
sns.histplot(merged_data['Weight'], kde=True)
plt.title('Weight Distribution')
plt.show()

# Calculate the correlation matrix
correlation_matrix = merged_data.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='Blues', cbar=True, square=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Encode the 'Gender' column with numerical values
merged_data['Gender'].replace({'male': 0, 'female': 1}, inplace=True)
print(merged_data.head())

# Define the feature matrix (X) and target vector (Y)
X = merged_data.drop(columns=['User_ID', 'Calories'])
Y = merged_data['Calories']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initialize the XGBoost Regressor model
xgb_model = XGBRegressor()

# Train the model on the training data
xgb_model.fit(X_train, Y_train)

# Predict the calorie values for the test set
predicted_calories = xgb_model.predict(X_test)
print(predicted_calories)

# Calculate the Mean Absolute Error (MAE) of the predictions
mae_value = metrics.mean_absolute_error(Y_test, predicted_calories)
print("Mean Absolute Error (MAE) = ", mae_value)
