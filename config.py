import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# 모델 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "solar-embedding-1-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "solar-pro")

# ChromaDB 설정
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "jeju_travel")

# 기타 설정
DEFAULT_NUM_RESULTS = int(os.getenv("DEFAULT_NUM_RESULTS", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# API 키 유효성 검사
if not UPSTAGE_API_KEY:
    print("⚠️ WARNING: UPSTAGE_API_KEY가 설정되지 않았습니다.")
    print("💡 .env 파일을 생성하고 API 키를 설정해주세요.")

# API 키 발급 안내
API_GUIDE = """
🔑 Upstage API 키 발급 방법:

1. Upstage Console (https://console.upstage.ai/) 접속
2. 회원가입 및 로그인
3. API 키 생성
4. .env 파일에 UPSTAGE_API_KEY=발급받은키 추가

💡 Solar Pro 모델 사용을 위해 API 키가 필요합니다.
""" 