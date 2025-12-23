import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from detectors.llm_detector import LLMDetector, FakeNewsResult


class TestLLMDetector:
    
    @pytest.fixture
    def mock_llm_response(self):
        return FakeNewsResult(
            is_fake=False,
            confidence=0.85,
            reasoning="Article appears credible with proper sources",
            red_flags=[]
        )
    
    @pytest.fixture
    def detector(self):
        with patch('detectors.llm_detector.ChatOpenAI') as mock_chat, \
             patch('detectors.llm_detector.ChatPromptTemplate') as mock_prompt, \
             patch('detectors.llm_detector.PydanticOutputParser') as mock_parser, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            
            # Mock chain
            mock_chain = MagicMock()
            mock_chain.invoke = MagicMock(return_value=FakeNewsResult(
                is_fake=False,
                confidence=0.85,
                reasoning="Test reasoning",
                red_flags=[]
            ))
            
            detector = LLMDetector()
            detector.chain = mock_chain
            return detector
    
    def test_detector_initialization_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                LLMDetector()
    
    def test_predict_success(self, detector):
        text = "This is a real news article."
        result = detector.predict(text)
        
        assert 'is_fake' in result
        assert 'confidence' in result
        assert 'label' in result
        assert 'reasoning' in result
        assert 'red_flags' in result
        assert 'method' in result
        assert result['method'] == 'llm'
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_empty_text(self, detector):
        with pytest.raises(ValueError, match="Empty text provided"):
            detector.predict("")
    
    def test_predict_error_handling(self, detector):
        detector.chain.invoke = MagicMock(side_effect=Exception("API Error"))
        
        result = detector.predict("Test text")
        
        assert result['error'] is True
        assert result['label'] == 'UNKNOWN'
        assert result['confidence'] == 0.5

