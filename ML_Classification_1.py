# First ML classification code

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

data = {
'CustomerID': range(1, 11),
'Age': [25, 34, 22, 45, 35, 50, 29, 41, 38, 30],
'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
'AnnualIncome': [50000, 60000, 45000, 80000, 75000, 90000, 62000, 82000, 70000, 65000],
'CreditScore': [700, 750, 710, 690, 720, 740, 705, 735, 710, 725],
'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
			
df = pd.DataFrame(data)
			
# Display the first few rows of the dataset
print(df.head())

# Encode Gender column
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
		
# Encode Purchased column
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})
		
# Display the first few rows of the encoded dataset
print(df.head())

# Features and target
X = df[['Age', 'Gender', 'AnnualIncome', 'CreditScore']]
y = df['Purchased']
		
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
# Display the training data
print("X_train:\n", X_train.head())
print("y_train:\n", y_train.head())


# Initialize the StandardScaler
scaler = StandardScaler()
		
# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)
		
# Transform the testing data
X_test_scaled = scaler.transform(X_test)
		
# Display the scaled training data
print("X_train_scaled:\n", X_train_scaled[:5])


# Initialize the logistic regression model
model = LogisticRegression()
		
# Train the model
model.fit(X_train_scaled, y_train)
		
# Display that the model has been trained
print("Model has been trained.")
# Make predictions on the test data
y_pred = model.predict(X_test_scaled)
		
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
		
# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
		
print("Model and scaler have been saved.")


# Load the model
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
		
# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
		
# Example new data
new_data = pd.DataFrame({
		    'Age': [28, 35],
		    'Gender': [0, 1],  # 0 for Male, 1 for Female
		    'AnnualIncome': [58000, 75000],
		    'CreditScore': [720, 710]
		})
		
# Standardize the new data using the loaded scaler
new_data_scaled = loaded_scaler.transform(new_data)
		
# Make predictions
new_predictions = loaded_model.predict(new_data_scaled)
		
print("Predictions:", new_predictions)


