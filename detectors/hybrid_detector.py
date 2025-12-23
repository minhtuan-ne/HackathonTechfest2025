import logging
from typing import Dict, Optional, Literal
from .ml_detector import MLDetector
from .llm_detector import LLMDetector

logger = logging.getLogger(__name__)


class HybridDetector:
    
    def __init__(
        self,
        ml_detector: Optional[MLDetector] = None,
        llm_detector: Optional[LLMDetector] = None,
        confidence_threshold: float = 0.7,
        fallback_to_llm: bool = True
    ):

        self.ml_detector = ml_detector or MLDetector()
        self.llm_detector = llm_detector
        self.confidence_threshold = confidence_threshold
        self.fallback_to_llm = fallback_to_llm
        
        # Try to initialize LLM detector if not provided
        if self.llm_detector is None and self.fallback_to_llm:
            try:
                self.llm_detector = LLMDetector()
                logger.info("LLM detector initialized successfully")
            except Exception as e:
                logger.warning(f"LLM detector not available: {e}. Will use ML only.")
                self.llm_detector = None
                self.fallback_to_llm = False
    
    def predict(
        self, 
        text: str, 
        method: Optional[Literal['ml', 'llm', 'auto']] = 'auto'
    ) -> Dict:
      
        if method == 'ml':
            return self.ml_detector.predict(text)
        
        if method == 'llm':
            if self.llm_detector is None:
                logger.warning("LLM detector not available, falling back to ML")
                return self.ml_detector.predict(text)
            return self.llm_detector.predict(text)
        
        # Auto mode: hybrid strategy
        try:
            # Step 1: Try ML first (fast)
            ml_result = self.ml_detector.predict(text)
            ml_confidence = ml_result['confidence']
            
            # Step 2: Decide if we need LLM
            needs_llm = (
                ml_confidence < self.confidence_threshold and 
                self.fallback_to_llm and 
                self.llm_detector is not None
            )
            
            if needs_llm:
                logger.debug(f"ML confidence ({ml_confidence:.2f}) below threshold, using LLM")
                llm_result = self.llm_detector.predict(text)
                
                # Combine results
                return {
                    **llm_result,
                    'ml_confidence': ml_confidence,
                    'ml_prediction': ml_result['label'],
                    'method': 'hybrid'
                }
            else:
                # ML is confident enough
                return ml_result
                
        except Exception as e:
            logger.error(f"Error in hybrid prediction: {e}", exc_info=True)
            # Fallback to ML if LLM fails
            if self.llm_detector is not None:
                try:
                    return self.llm_detector.predict(text)
                except:
                    pass
            raise
    
    def get_stats(self) -> Dict:
        """Get detector statistics and availability"""
        return {
            'ml_available': self.ml_detector is not None,
            'llm_available': self.llm_detector is not None,
            'confidence_threshold': self.confidence_threshold,
            'fallback_enabled': self.fallback_to_llm
        }

