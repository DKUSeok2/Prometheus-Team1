"""
기본 에이전트 클래스
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_community.llms import Ollama
from ..config import settings

class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> Ollama:
        """Ollama LLM 초기화"""
        return Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.7
        )
    
    @abstractmethod
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에이전트 처리 로직"""
        pass
    
    def format_response(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """응답 포맷팅"""
        return {
            "agent": self.name,
            "content": content,
            "metadata": metadata or {}
        }
