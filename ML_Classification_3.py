"""
Name Gender Prediction

This script trains Logistic Regression and Random Forest Classifier models to predict gender based on names.
It also demonstrates saving and loading these trained models using pickle, and making predictions with the loaded models.

Steps:
1. Load dataset containing names labeled with genders (male and female) from a CSV file.
2. Split dataset into training and testing sets.
3. Convert names into numerical features using CountVectorizer.
4. Train a Logistic Regression model and save it to a pickle file.
5. Train a Random Forest Classifier model and save it to a pickle file.
6. Evaluate both models on the testing set and print their accuracies.
7. Load the trained models from pickle files.
8. Make predictions on example names using the loaded models and print results.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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

# Step 4: Train a Logistic Regression model and save it to a pickle file
logreg_model = LogisticRegression()
logreg_model.fit(X_train_vectorized, y_train)
logreg_model_file = 'logreg_model.pkl'
with open(logreg_model_file, 'wb') as file:
    pickle.dump(logreg_model, file)

# Step 5: Train a Random Forest Classifier model and save it to a pickle file
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train_vectorized, y_train)
rf_model_file = 'rf_model.pkl'
with open(rf_model_file, 'wb') as file:
    pickle.dump(rf_model, file)

# Step 6: Evaluate models
logreg_predictions = logreg_model.predict(X_test_vectorized)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print(f"Logistic Regression Accuracy: {logreg_accuracy}")

rf_predictions = rf_model.predict(X_test_vectorized)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Step 7: Load the trained models from pickle files
with open(logreg_model_file, 'rb') as file:
    loaded_logreg_model = pickle.load(file)

with open(rf_model_file, 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Step 8: Make predictions with loaded models on example names
names_to_predict = ["Alice", "Bob", "Emma", "James"]
names_vectorized = vectorizer.transform(names_to_predict)

logreg_predictions_loaded = loaded_logreg_model.predict(names_vectorized)
rf_predictions_loaded = loaded_rf_model.predict(names_vectorized)

for name, logreg_pred, rf_pred in zip(names_to_predict, logreg_predictions_loaded, rf_predictions_loaded):
    print(f"Name: {name}, Logistic Regression Prediction: {logreg_pred}, Random Forest Prediction: {rf_pred}")

