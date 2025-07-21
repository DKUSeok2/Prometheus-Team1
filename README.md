# 🤖 ORDA AI Engine

**제주도 여행 AI 챗봇 '오르다'의 핵심 AI 엔진**

> 이 브랜치는 [오르다 메인 프로젝트](https://github.com/DKUSeok2/Prometheus-Team1)의 AI 엔진 모음입니다.  
> 다양한 LLM과 RAG 기술을 활용한 챗봇 구현체들과 데이터 처리 도구들을 포함합니다.

## 📋 목차
- [🚀 빠른 시작](#-빠른-시작)
- [🤖 챗봇 종류](#-챗봇-종류)
- [🛠️ 기술 스택](#️-기술-스택)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🔧 설치 및 실행](#-설치-및-실행)
- [🔌 API 서비스](#-api-서비스)

## 🚀 빠른 시작

```bash
# 1. 저장소 클론 (AI 브랜치)
git clone -b AI https://github.com/DKUSeok2/Prometheus-Team1.git
cd Prometheus-Team1

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 설정
cp config.example.py config.py
# config.py에서 API 키들 설정

# 4. 데이터 준비 (선택사항)
python data_loader.py

# 5. 챗봇 실행 (여러 옵션)
python final_chatbot.py        # 최종 버전 (추천)
python chatbot_service.py      # API 서비스
python interactive_demo.py     # 대화형 데모
```

## 🤖 챗봇 종류

### 🎯 **최종 버전 (권장)**
- **final_chatbot.py**: LangGraph 기반 Multi-agent 챗봇
  - 컨텍스트 인식 대화
  - 사용자 프로필 관리
  - 조건부 프롬프트 (일정 생성 vs 일반 추천)

- **interactive_demo.py**: 프롬프트 엔지니어링 데모
  - 대화 테스트 환경
  - 컨텍스트 연결 테스트 (`--test-context`)

### 🔌 **API 서비스**
- **chatbot_service.py**: FastAPI 기반 챗봇 서비스
  - REST API 엔드포인트 제공
  - 백엔드와 연동 가능

### 🛠️ **데이터 도구**
- **data_loader.py**: 제주도 여행 데이터 로딩 및 ChromaDB 초기화

## 🛠️ 기술 스택

### 🧠 **LLM & Embedding**
- **Solar Pro** (Upstage): 대화 생성
- **Solar Embedding** (Upstage): 벡터 임베딩  
- **Llama 3.2 1B**: 경량 모델 옵션
- **HuggingFace Transformers**: 다양한 모델 지원

### 🔍 **RAG & Vector DB**
- **ChromaDB**: 벡터 데이터베이스
- **LangChain**: RAG 파이프라인
- **Sentence Transformers**: 임베딩 처리

### 🤖 **Agent Framework**
- **LangGraph**: Multi-agent 오케스트레이션
- **Multi-agent System**: 역할 기반 에이전트 분리

### 🌐 **API & Service**
- **FastAPI**: REST API 서비스
- **HTTPX**: 비동기 HTTP 클라이언트
- **Uvicorn**: ASGI 서버

## 📁 프로젝트 구조

```
AI/
├── README.md                    # 프로젝트 가이드
├── requirements.txt             # Python 의존성
├── .gitignore                   # 보안 설정
├── config.example.py            # 설정 파일 템플릿
│
├── 🎯 핵심 AI 파일들
│   ├── final_chatbot.py         # 메인 챗봇 (LangGraph Multi-agent)
│   ├── interactive_demo.py      # 프롬프트 엔지니어링 데모
│   ├── chatbot_service.py       # FastAPI 서비스 (백엔드 연동)
│   └── data_loader.py           # 데이터 로딩 유틸리티
│
└── data/                        # 제주도 여행 데이터
    ├── visitjeju_event.json     # 행사 정보
    ├── visitjeju_food.json      # 맛집 정보
    ├── visitjeju_hotel.json     # 숙소 정보
    └── visitjeju_tour.json      # 관광지 정보
```

## 🔧 설치 및 실행

### 1. 환경 준비
- Python 3.8 이상
- 필요한 API 키들 (Upstage, HuggingFace 등)

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 설정 파일 준비
```bash
cp config.example.py config.py
```

`config.py`에서 다음 API 키들을 설정하세요:
- `UPSTAGE_API_KEY`: [Upstage Console](https://console.upstage.ai/)에서 발급
- `HUGGINGFACE_TOKEN`: [HuggingFace](https://huggingface.co/settings/tokens)에서 발급
- `TOUR_API_KEY`: [공공데이터포털](https://data.go.kr/)에서 발급 (선택사항)

### 4. 데이터 초기화 (선택사항)
```bash
python data_loader.py
```

### 5. 챗봇 실행

#### 🎯 최종 버전 (추천)
```bash
# 메인 챗봇 실행
python final_chatbot.py

# 대화형 데모 (프롬프트 테스트용)
python interactive_demo.py

# 컨텍스트 테스트
python interactive_demo.py --test-context
```

#### 🔌 API 서비스
```bash
# 챗봇 API 서비스 실행 (포트 8001)
python chatbot_service.py
```

## 🔌 API 서비스

### 챗봇 서비스 API
`chatbot_service.py`를 실행하면 `http://localhost:8001`에서 다음 API를 제공합니다:

```bash
# 채팅 요청
POST http://localhost:8001/chat
{
  "content": "제주도 맛집 추천해줘",
  "session_id": "user123",
  "conversation_history": [...],
  "user_profile": {...}
}

# 헬스 체크
GET http://localhost:8001/health
```

### 백엔드 연동
이 API 서비스는 [Backend 브랜치](https://github.com/DKUSeok2/Prometheus-Team1/tree/Backend)의 FastAPI 서버와 연동됩니다:

```
Backend API ↔ Chatbot Service ↔ LLM/RAG
    :8000         :8001          Models
```

## 🔗 메인 프로젝트 연관성

### 📂 전체 오르다 프로젝트 구조
- **메인 브랜치**: Streamlit 기반 웹 애플리케이션
- **Prompt 브랜치**: 프롬프트 엔지니어링 데모  
- **Backend 브랜치**: FastAPI 백엔드 서버
- **AI 브랜치** (현재): 챗봇 AI 엔진 모음

### 🔄 개발 플로우
1. **AI 브랜치**에서 챗봇 알고리즘 개발/실험
2. **Prompt 브랜치**에서 프롬프트 최적화
3. **Backend 브랜치**에서 API 서버 개발
4. **메인 브랜치**에서 전체 앱 통합

## 📊 핵심 파일 역할

| 파일명 | 용도 | 실행 방법 | 특징 |
|--------|------|-----------|------|
| **final_chatbot.py** | 메인 챗봇 | `python final_chatbot.py` | 프로덕션용, 컨텍스트 인식 |
| **interactive_demo.py** | 프롬프트 테스트 | `python interactive_demo.py` | 프롬프트 엔지니어링 환경 |
| **chatbot_service.py** | API 서비스 | `python chatbot_service.py` | 백엔드 연동용 REST API |
| **data_loader.py** | 데이터 초기화 | `python data_loader.py` | ChromaDB 벡터 데이터베이스 구축 |

## 🚨 주의사항

### API 사용량 관리
- **Solar Pro API**: 유료 서비스, 사용량 모니터링 필요
- **HuggingFace**: 무료 Tier 제한 확인
- **공공데이터포털**: API 호출량 제한 확인

### 모델 다운로드
- 일부 HuggingFace 모델들은 초기 실행 시 자동 다운로드
- 충분한 디스크 공간 확보 (모델당 1-4GB)

## 🆘 문제 해결

### 자주 발생하는 오류

#### 1. API 키 오류
```
AuthenticationError: Invalid API key
```
**해결책**: config.py에서 올바른 API 키 설정 확인

#### 2. ChromaDB 초기화 오류  
```
chromadb.errors.InvalidDimensionException
```
**해결책**: `python data_loader.py` 재실행

#### 3. 메모리 부족
```
CUDA out of memory / RAM shortage
```
**해결책**: 더 작은 모델 사용 (Llama-3.2-1B 등)

#### 4. 모델 다운로드 실패
```
ConnectionError: Unable to download model
```
**해결책**: 네트워크 연결 확인, HuggingFace 토큰 설정

## 📞 지원

- **기술적 문제**: [GitHub Issues](https://github.com/DKUSeok2/Prometheus-Team1/issues)
- **메인 프로젝트**: [Prometheus Team1 Repository](https://github.com/DKUSeok2/Prometheus-Team1)
- **AI 모델 문의**: Upstage Console, HuggingFace Community

## 🤝 기여 방법

### 프롬프트 개선
1. `interactive_demo.py`로 테스트
2. 개선된 프롬프트를 `final_chatbot.py`에 적용
3. 성능 평가 결과 공유

### 챗봇 기능 개선
1. `final_chatbot.py`에서 새로운 기능 추가
2. `chatbot_service.py`의 API도 함께 업데이트
3. 테스트 후 Pull Request 생성

---

**🤖 다양한 AI 기술로 더 똑똑한 제주도 여행 챗봇을 만들어나갑니다!**

**지속적인 실험과 개선을 통해 최고의 여행 AI 어시스턴트 '오르다'를 완성해요! 🌴✨** 