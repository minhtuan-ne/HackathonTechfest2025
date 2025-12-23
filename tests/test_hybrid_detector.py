import pytest
from unittest.mock import Mock, MagicMock
from detectors.hybrid_detector import HybridDetector
from detectors.ml_detector import MLDetector


class TestHybridDetector:
    
    @pytest.fixture
    def mock_ml_detector(self):
        """Mock ML detector"""
        ml = Mock(spec=MLDetector)
        ml.predict = MagicMock(return_value={
            'is_fake': False,
            'confidence': 0.8,
            'label': 'REAL',
            'method': 'ml',
            'text_length': 50
        })
        return ml
    
    @pytest.fixture
    def mock_llm_detector(self):
        llm = Mock()
        llm.predict = MagicMock(return_value={
            'is_fake': False,
            'confidence': 0.9,
            'label': 'REAL',
            'reasoning': 'Test reasoning',
            'red_flags': [],
            'method': 'llm',
            'text_length': 50
        })
        return llm
    
    @pytest.fixture
    def hybrid_detector(self, mock_ml_detector, mock_llm_detector):
        return HybridDetector(
            ml_detector=mock_ml_detector,
            llm_detector=mock_llm_detector,
            confidence_threshold=0.7,
            fallback_to_llm=True
        )
    
    def test_predict_ml_only(self, hybrid_detector, mock_ml_detector):
        result = hybrid_detector.predict("Test text", method='ml')
        
        mock_ml_detector.predict.assert_called_once_with("Test text")
        assert result['method'] == 'ml'
    
    def test_predict_llm_only(self, hybrid_detector, mock_llm_detector):
        result = hybrid_detector.predict("Test text", method='llm')
        
        mock_llm_detector.predict.assert_called_once_with("Test text")
        assert result['method'] == 'llm'
    
    def test_predict_auto_high_confidence(self, hybrid_detector, mock_ml_detector, mock_llm_detector):
        mock_ml_detector.predict.return_value = {
            'is_fake': False,
            'confidence': 0.9,  # High confidence
            'label': 'REAL',
            'method': 'ml'
        }
        
        result = hybrid_detector.predict("Test text", method='auto')
        
        mock_ml_detector.predict.assert_called_once()
        mock_llm_detector.predict.assert_not_called()
        assert result['method'] == 'ml'
    
    def test_predict_auto_low_confidence(self, hybrid_detector, mock_ml_detector, mock_llm_detector):
        mock_ml_detector.predict.return_value = {
            'is_fake': False,
            'confidence': 0.5,  # Low confidence
            'label': 'REAL',
            'method': 'ml'
        }
        
        result = hybrid_detector.predict("Test text", method='auto')
        
        mock_ml_detector.predict.assert_called_once()
        mock_llm_detector.predict.assert_called_once()
        assert result['method'] == 'hybrid'
        assert 'ml_confidence' in result
    
    def test_get_stats(self, hybrid_detector):
        stats = hybrid_detector.get_stats()
        
        assert 'ml_available' in stats
        assert 'llm_available' in stats
        assert 'confidence_threshold' in stats
        assert 'fallback_enabled' in stats
        assert stats['ml_available'] is True
        assert stats['llm_available'] is True

