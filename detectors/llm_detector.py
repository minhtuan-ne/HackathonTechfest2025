import os
import logging
from typing import Dict, Optional
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain openai langchain-openai")

logger = logging.getLogger(__name__)


class FakeNewsResult(BaseModel):
    """Structured output schema for LLM predictions"""
    is_fake: bool = Field(description="True if the article is fake news")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the decision")
    red_flags: list[str] = Field(description="List of suspicious elements found", default_factory=list)


class LLMDetector:
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        api_key: Optional[str] = None
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain openai"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key
        )
        
        # Create output parser
        self.output_parser = PydanticOutputParser(pydantic_object=FakeNewsResult)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert fact-checker and news analyst. Analyze news articles for signs of fake news.

Consider these factors:
- Writing quality and style (grammar, coherence, professionalism)
- Sensationalism and emotional manipulation (excessive exclamation, fear-mongering)
- Factual claims and verifiability (specific claims that can be checked)
- Source credibility indicators (authority, transparency)
- Logical consistency and coherence (does the story make sense?)
- Common fake news patterns (clickbait headlines, conspiracy theories, etc.)

Provide a clear, objective analysis.

{format_instructions}"""),
            ("human", "Analyze this article and determine if it's fake news:\n\n{article}")
        ])
        
        # Create chain using LangChain's LCEL syntax
        self.chain = self.prompt | self.llm | self.output_parser
        
        logger.info(f"LLM Detector initialized with model: {model_name}")
    
    def predict(self, text: str) -> Dict:
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided")
        
        try:
            # Invoke chain
            result = self.chain.invoke({
                "article": text,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            return {
                "is_fake": result.is_fake,
                "confidence": result.confidence,
                "label": "FAKE" if result.is_fake else "REAL",
                "reasoning": result.reasoning,
                "red_flags": result.red_flags,
                "method": "llm",
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in LLM prediction: {e}", exc_info=True)
            # Return fallback result
            return {
                "is_fake": False,
                "confidence": 0.5,
                "label": "UNKNOWN",
                "reasoning": f"Error during analysis: {str(e)}",
                "red_flags": [],
                "method": "llm",
                "text_length": len(text),
                "error": True
            }

