import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app) 

clf = None
vectorizer = None

# Train the model
def train_and_save_model():
    global clf, vectorizer
    
    print("Loading dataset and training new model...")
    
    data = pd.read_csv("fake_or_real_news.csv")
    data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    data = data.drop("label", axis=1)
    x, y = data['text'], data['fake']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    clf = LinearSVC()
    clf.fit(x_train_vectorized, y_train)
    
    accuracy = clf.score(x_test_vectorized, y_test)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    model_data = {
        'classifier': clf,
        'vectorizer': vectorizer
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved to model.pkl")
    return clf, vectorizer

def load_model():
    global clf, vectorizer
    
    if os.path.exists('model.pkl'):
        print("Loading existing model...")
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        clf = model_data['classifier']
        vectorizer = model_data['vectorizer']
        print("Model loaded successfully")
        return True
    else:
        print("No existing model found")
        return False

if not load_model():
    clf, vectorizer = train_and_save_model()

# Fetch the API to front end
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        
        text_vectorized = vectorizer.transform([text])
        
        prediction = clf.predict(text_vectorized)[0]
        
        confidence_score = clf.decision_function(text_vectorized)[0]
        
        confidence = min(max(abs(confidence_score) / 2, 0), 1)
        
        result = {
            'is_fake': bool(prediction),
            'confidence': float(confidence),
            'label': 'FAKE' if prediction == 1 else 'REAL',
            'text_length': len(text)
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Retrain the model
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        clf, vectorizer = train_and_save_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)