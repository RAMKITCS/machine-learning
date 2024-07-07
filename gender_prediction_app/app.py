from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model, vectorizer, and scaler
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

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

@app.route('/', methods=['GET', 'POST'])
def home():
    gender = ''
    confidence = 0.0
    if request.method == 'POST':
        name = request.form['name']
        features = name_features(name)
        vectorized_features = vectorizer.transform([features])
        scaled_features = scaler.transform(vectorized_features)
        prediction = model.predict(scaled_features)
        confidence_score = model.predict_proba(scaled_features)
        gender = 'Male' if prediction[0] == 0 else 'Female'
        confidence = confidence_score[0][prediction[0]] * 100
    return render_template('index.html', gender=gender, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
