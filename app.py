import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Import detectors
from detectors import MLDetector, LLMDetector, HybridDetector

app = Flask(__name__)
CORS(app)

# Initialize detectors (lazy loading)
ml_detector = None
llm_detector = None
hybrid_detector = None


def get_ml_detector():
    """Get or create ML detector"""
    global ml_detector
    if ml_detector is None:
        ml_detector = MLDetector()
    return ml_detector


def get_llm_detector():
    """Get or create LLM detector (if API key available)"""
    global llm_detector
    if llm_detector is None:
        try:
            llm_detector = LLMDetector()
        except Exception as e:
            logger.warning(f"LLM detector not available: {e}")
            return None
    return llm_detector


def get_hybrid_detector():
    """Get or create hybrid detector"""
    global hybrid_detector
    if hybrid_detector is None:
        ml = get_ml_detector()
        llm = get_llm_detector()
        hybrid_detector = HybridDetector(
            ml_detector=ml,
            llm_detector=llm,
            confidence_threshold=float(os.environ.get('HYBRID_CONFIDENCE_THRESHOLD', 0.7)),
            fallback_to_llm=os.environ.get('FALLBACK_TO_LLM', 'true').lower() == 'true'
        )
    return hybrid_detector


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    ml = get_ml_detector()
    llm = get_llm_detector()
    
    return jsonify({
        'status': 'healthy',
        'ml_available': ml is not None,
        'llm_available': llm is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if text is fake news
    
    Request body:
        {
            "text": "article text...",
            "method": "auto" | "ml" | "llm"  // optional, defaults to "auto"
        }
    
    Response:
        {
            "is_fake": bool,
            "confidence": float,
            "label": "FAKE" | "REAL",
            "method": "ml" | "llm" | "hybrid",
            "reasoning": str,  // only for LLM/hybrid
            "red_flags": list,  // only for LLM/hybrid
            "text_length": int
        }
    """
    try:
        try:
            data = request.get_json()
        except Exception as e:
            logger.warning(f"Invalid JSON: {e}")
            return jsonify({'error': 'Invalid JSON provided'}), 400
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get method (auto, ml, or llm)
        method = data.get('method', 'auto').lower()
        
        if method == 'ml':
            detector = get_ml_detector()
            result = detector.predict(text)
        elif method == 'llm':
            detector = get_llm_detector()
            if detector is None:
                return jsonify({
                    'error': 'LLM detector not available. Set OPENAI_API_KEY environment variable.'
                }), 503
            result = detector.predict(text)
        else:  # auto - use hybrid
            detector = get_hybrid_detector()
            result = detector.predict(text, method='auto')
        
        return jsonify(result)
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your request'}), 500


@app.route('/predict/ml', methods=['POST'])
def predict_ml():
    """Force ML-only prediction"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        detector = get_ml_detector()
        result = detector.predict(data['text'])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict/llm', methods=['POST'])
def predict_llm():
    """Force LLM-only prediction"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    detector = get_llm_detector()
    if detector is None:
        return jsonify({
            'error': 'LLM detector not available. Set OPENAI_API_KEY environment variable.'
        }), 503
    
    try:
        result = detector.predict(data['text'])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in LLM prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the ML model"""
    try:
        # Check for API key if required
        api_key = request.headers.get('X-API-Key') or request.json.get('api_key')
        expected_key = os.environ.get('RETRAIN_API_KEY')
        
        if expected_key and api_key != expected_key:
            return jsonify({'error': 'Unauthorized'}), 401
        
        detector = get_ml_detector()
        result = detector.retrain()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error retraining model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get detector statistics"""
    hybrid = get_hybrid_detector()
    return jsonify(hybrid.get_stats())


@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    Get model evaluation metrics
    
    Returns comprehensive evaluation metrics including:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion matrix
    - Classification report
    """
    try:
        detector = get_ml_detector()
        
        # Load test data
        import pandas as pd
        data_path = os.environ.get('DATA_PATH', 'fake_or_real_news.csv')
        data = pd.read_csv(data_path)
        data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        x, y = data['text'], data['fake']
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Predict on test set
        predictions = []
        for text in x_test:
            try:
                result = detector.predict(text)
                predictions.append(1 if result['is_fake'] else 0)
            except:
                predictions.append(0)  # Fallback
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions).tolist()
        
        return jsonify({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'true_negative': cm[0][0],
                'false_positive': cm[0][1],
                'false_negative': cm[1][0],
                'true_positive': cm[1][1]
            },
            'test_samples': len(y_test),
            'model_type': 'ML (TF-IDF + LinearSVC)'
        })
    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(debug=debug, host='0.0.0.0', port=port)

