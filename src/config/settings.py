"""
환경 설정 관리 모듈
"""
import os
from dotenv import load_dotenv
from typing import Optional

class Settings:
    """애플리케이션 설정 클래스"""
    
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        # API Keys
        self.upstage_api_key = self._get_env_var("UPSTAGE_API_KEY")
        self.kakao_api_key = self._get_env_var("KAKAO_API_KEY", required=False)  # REST API 키
        self.kakao_js_key = self._get_env_var("KAKAO_JS_KEY", required=False)    # JavaScript 키
        
        # Amadeus 항공권 API
        self.amadeus_client_id = self._get_env_var("AMADEUS_CLIENT_ID", required=False)
        self.amadeus_client_secret = self._get_env_var("AMADEUS_CLIENT_SECRET", required=False)
        self.amadeus_base_url = self._get_env_var("AMADEUS_BASE_URL", "https://test.api.amadeus.com")
        
        # Ollama 설정
        self.ollama_base_url = self._get_env_var("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = self._get_env_var("OLLAMA_MODEL", "gemma3:4b")
        
        # ChromaDB 설정
        self.chromadb_path = self._get_env_var("CHROMADB_PATH", "./chroma_db")
        self.collection_name = self._get_env_var("COLLECTION_NAME", "visitjeju")
        
        # 임베딩 모델 설정
        self.query_embedding_model = "solar-embedding-1-large-query"
        self.passage_embedding_model = "solar-embedding-1-large-passage"
        
        # 앱 설정
        self.app_title = "오르다 - 제주도 여행 추천 챗봇"
        self.max_results = int(self._get_env_var("MAX_RESULTS", "5"))
        self.chat_history_path = self._get_env_var("CHAT_HISTORY_PATH", "./chat_history")
        
    def _get_env_var(self, key: str, default: Optional[str] = None, required: bool = True) -> str:
        """환경 변수 가져오기"""
        value = os.getenv(key, default)
        if required and not value:
            raise ValueError(f"환경 변수 {key}가 설정되어 있지 않습니다.")
        return value
    
    def validate_settings(self) -> bool:
        """설정 유효성 검사"""
        try:
            # 필수 API 키 확인
            if not self.upstage_api_key:
                raise ValueError("UPSTAGE_API_KEY가 필요합니다.")
            
            # ChromaDB 경로 생성
            os.makedirs(self.chromadb_path, exist_ok=True)
            os.makedirs(self.chat_history_path, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"설정 검증 실패: {e}")
            return False

# 전역 설정 인스턴스
settings = Settings()