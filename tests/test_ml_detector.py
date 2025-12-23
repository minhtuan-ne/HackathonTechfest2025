import pytest
import os
import tempfile
import shutil
from detectors.ml_detector import MLDetector


class TestMLDetector:
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_data_file(self, temp_dir):
        import pandas as pd
        
        data = {
            'text': [
                'This is a real news article with factual information and proper sources.',
                'BREAKING: SHOCKING NEWS!!! This will blow your mind!!! Click now!!!',
                'According to official sources, the event occurred yesterday.',
                'ALIENS INVADE EARTH!!! Government hiding the truth!!!',
            ],
            'label': ['REAL', 'FAKE', 'REAL', 'FAKE']
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def detector(self, temp_dir, sample_data_file):
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        return MLDetector(model_path=model_path, data_path=sample_data_file)
    
    def test_detector_initialization(self, detector):
        assert detector.clf is not None
        assert detector.vectorizer is not None
    
    def test_predict_real_news(self, detector):
        text = "This is a well-written article with proper sources and factual information."
        result = detector.predict(text)
        
        assert 'is_fake' in result
        assert 'confidence' in result
        assert 'label' in result
        assert 'method' in result
        assert 'text_length' in result
        assert result['method'] == 'ml'
        assert isinstance(result['is_fake'], bool)
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_fake_news(self, detector):
        text = "BREAKING: SHOCKING NEWS!!! This will blow your mind!!!"
        result = detector.predict(text)
        
        assert 'is_fake' in result
        assert 'confidence' in result
        assert result['label'] in ['FAKE', 'REAL']
        assert result['text_length'] == len(text)
    
    def test_predict_empty_text(self, detector):
        with pytest.raises(ValueError, match="Empty text provided"):
            detector.predict("")
        
        with pytest.raises(ValueError, match="Empty text provided"):
            detector.predict("   ")
    
    def test_predict_whitespace_only(self, detector):
        with pytest.raises(ValueError):
            detector.predict("\n\t   \n")
    
    def test_confidence_range(self, detector):
        texts = [
            "Real news article with proper sources.",
            "FAKE NEWS!!! CLICKBAIT!!!",
            "A normal article about current events.",
        ]
        
        for text in texts:
            result = detector.predict(text)
            assert 0 <= result['confidence'] <= 1
    
    def test_retrain(self, detector):
        result = detector.retrain()
        assert result['status'] == 'success'
        assert 'message' in result
    
    def test_model_persistence(self, temp_dir, sample_data_file):
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        # Create and train detector
        detector1 = MLDetector(model_path=model_path, data_path=sample_data_file)
        text = "Test article"
        result1 = detector1.predict(text)
        
        # Create new detector (should load existing model)
        detector2 = MLDetector(model_path=model_path, data_path=sample_data_file)
        result2 = detector2.predict(text)
        
        # Results should be identical
        assert result1['is_fake'] == result2['is_fake']
        assert abs(result1['confidence'] - result2['confidence']) < 0.01

