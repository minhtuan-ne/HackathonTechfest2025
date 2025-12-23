import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MLDetector:
    
    def __init__(self, model_path: Optional[str] = None, data_path: Optional[str] = None):
        self.model_path = model_path or os.environ.get('MODEL_PATH', 'model.pkl')
        self.data_path = data_path or os.environ.get('DATA_PATH', 'fake_or_real_news.csv')
        self.clf = None
        self.vectorizer = None
        self._load_or_train()
    
    def _load_or_train(self):
        if self._load_model():
            logger.info("ML model loaded successfully")
        else:
            logger.info("No existing model found, training new model...")
            self._train_model()
            logger.info("ML model trained and saved")
    
    def _load_model(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.clf = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
    
    def _train_model(self):
        # Load data
        data = pd.read_csv(self.data_path)
        data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
        data = data.drop("label", axis=1)
        x, y = data['text'], data['fake']
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
        x_train_vectorized = self.vectorizer.fit_transform(x_train)
        x_test_vectorized = self.vectorizer.transform(x_test)
        
        # Train classifier
        self.clf = LinearSVC(random_state=42)
        self.clf.fit(x_train_vectorized, y_train)
        
        # Evaluate
        y_pred = self.clf.predict(x_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        logger.info(f"Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.debug(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        model_data = {
            'classifier': self.clf,
            'vectorizer': self.vectorizer
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def predict(self, text: str) -> Dict:
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided")
        
        if self.clf is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded. Please train or load model first.")
        
        # Vectorize and predict
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.clf.predict(text_vectorized)[0]
        
        # Calculate confidence from decision function
        decision_score = self.clf.decision_function(text_vectorized)[0]
        # Normalize decision score to [0, 1] using sigmoid
        confidence = 1 / (1 + np.exp(-decision_score))
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'is_fake': bool(prediction),
            'confidence': float(confidence),
            'label': 'FAKE' if prediction == 1 else 'REAL',
            'method': 'ml',
            'text_length': len(text)
        }
    
    def retrain(self):
        """Retrain the model with current data"""
        self._train_model()
        return {'status': 'success', 'message': 'Model retrained successfully'}

