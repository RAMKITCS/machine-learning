#ML Classification to predict a name if its Male or Female

import nltk
nltk.download('names')
from nltk.corpus import names
import random
import string
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
def load_data():
    male_names = [(name, 'male') for name in names.words('male.txt')]
    female_names = [(name, 'female') for name in names.words('female.txt')]
    labeled_names = male_names + female_names
    random.shuffle(labeled_names)
    return labeled_names

# Feature extraction: Extract features from names
def name_features(name):
    return {
        'last_letter': name[-1].lower(),
        'last_two': name[-2:].lower(),
        'last_three': name[-3:].lower(),
        'first_letter': name[0].lower(),
        'first_two': name[:2].lower(),
        'first_three': name[:3].lower(),
        'length': len(name),
        'vowel_count': sum(1 for char in name if char.lower() in 'aeiou')
    }

# Prepare the data
labeled_names = load_data()
features = [name_features(name) for name, gender in labeled_names]
labels = [gender for name, gender in labeled_names]

# Vectorize features
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(features)
y = pd.Series(labels).map({'male': 0, 'female': 1}).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the vectorizer, scaler, and model
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['male', 'female'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Load and use the vectorizer, scaler, and model for new predictions
with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Predict gender of a new name
def predict_gender(name):
    features = name_features(name)
    vectorized_features = loaded_vectorizer.transform([features])
    scaled_features = loaded_scaler.transform(vectorized_features)
    prediction = loaded_model.predict(scaled_features)
    return 'male' if prediction[0] == 0 else 'female'

# Example prediction
new_name = 'Shradha'
predicted_gender = predict_gender(new_name)
print(f'The name {new_name} is predicted to be {predicted_gender}.')
