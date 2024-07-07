# ML Classification using Random forest to predict gender based on name
"""
Random Forest Classifier for Gender Prediction Based on Names

This script trains a Random Forest Classifier to predict gender (male or female) based on names.
It demonstrates how to:
1. Load a dataset containing names labeled with genders from a CSV file.
2. Split the dataset into training and testing sets.
3. Convert names into numerical features using CountVectorizer.
4. Train a Random Forest Classifier model.
5. Evaluate the model on the testing set and print its accuracy.
6. Save the trained model to a pickle file.
7. Load the trained model from the pickle file.
8. Make predictions on example names using the loaded model and print results.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
data = pd.read_csv('names.csv')

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['name'], data['gender'], test_size=0.2, random_state=0)

# Step 3: Convert names to features using CountVectorizer
vectorizer = CountVectorizer(analyzer='char', lowercase=False)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 4: Train a Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train_vectorized, y_train)

# Step 5: Evaluate the model
rf_predictions = rf_model.predict(X_test_vectorized)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Step 6: Save the trained model to a pickle file
rf_model_file = 'rf_model.pkl'
with open(rf_model_file, 'wb') as file:
    pickle.dump(rf_model, file)

# Step 7: Load the trained model from the pickle file
with open(rf_model_file, 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Step 8: Make predictions with the loaded model on example names
names_to_predict = ["Alice", "Bob", "Emma", "James"]
names_vectorized = vectorizer.transform(names_to_predict)

rf_predictions_loaded = loaded_rf_model.predict(names_vectorized)

# Output predictions
for name, prediction in zip(names_to_predict, rf_predictions_loaded):
    print(f"Name: {name}, Predicted Gender: {prediction}")