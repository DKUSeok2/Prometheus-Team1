# API 키 및 모델 설정 파일

# Upstage API 키 (임베딩용)
UPSTAGE_API_KEY = "your_upstage_api_key_here"

# 모델 설정
EMBEDDING_MODEL = "solar-embedding-1-large"  # Upstage 임베딩
CHAT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Llama 3.2 1B 대화 모델 (1-2GB)

# ChromaDB 설정
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "visitjeju_travel"

# 기타 설정
DEFAULT_NUM_RESULTS = 5
MAX_TOKENS = 1024

# 한국관광공사 Tour API 키 (향후 사용)
TOUR_API_KEY = "your_tour_api_key_here"

# HuggingFace API 키 (Gemma 모델용)
HUGGINGFACE_TOKEN = "your_huggingface_token_here"

# OpenAI API 키 (백업용)
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"

# 제주도 지역 코드
JEJU_AREA_CODE = "39"

# API 키 발급 안내
API_GUIDE = """
🔑 API 키 발급 방법:

1. 공공데이터포털 (https://data.go.kr) 회원가입
2. "국문 관광정보 서비스" 검색
3. "한국관광공사_국문 관광정보 서비스_GW" 활용신청
4. 발급받은 API 키를 TOUR_API_KEY에 입력
5. python tour_api_collector.py 실행

💡 승인까지 보통 1-2시간 소요됩니다.
""" 
