# 🌴 오르다(ORDA) - 프롬프트 엔지니어링 데모

**제주도 여행 AI 챗봇 '오르다'의 프롬프트 개선 및 테스트 환경**

> 이 브랜치는 [오르다 메인 프로젝트](https://github.com/DKUSeok2/Prometheus-Team1)의 프롬프트 엔지니어링 전용 데모입니다.  
> 팀원들이 챗봇의 응답 품질을 개선하기 위한 프롬프트 튜닝 작업을 수행할 수 있습니다.

## 📋 목차
- [🚀 빠른 시작](#-빠른-시작)
- [🛠️ 설치 방법](#️-설치-방법)
- [💬 사용법](#-사용법)
- [🎯 프롬프트 엔지니어링 가이드](#-프롬프트-엔지니어링-가이드)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🔧 설정 방법](#-설정-방법)

## 🚀 빠른 시작

```bash
# 1. 저장소 클론 (Prompt 브랜치)
git clone -b Prompt https://github.com/DKUSeok2/Prometheus-Team1.git
cd Prometheus-Team1

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
# .env 파일 생성하고 API 키 설정
echo "UPSTAGE_API_KEY=your_upstage_api_key_here" > .env
echo "EMBEDDING_MODEL=solar-embedding-1-large" >> .env
echo "CHROMA_DB_PATH=./chroma_db" >> .env
echo "COLLECTION_NAME=jeju_travel" >> .env

# 4. 데이터베이스 초기화
python setup_database.py

# 5. 인터랙티브 데모 실행
python interactive_demo.py
```

## 🛠️ 설치 방법

### 사전 요구사항
- Python 3.8 이상
- [Upstage API 키](https://console.upstage.ai/) (Solar Pro 모델 사용)

### 1. 가상환경 생성 (권장)
```bash
python -m venv jeju_chatbot
source jeju_chatbot/bin/activate  # Windows: jeju_chatbot\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:
```env
UPSTAGE_API_KEY=your_upstage_api_key_here
EMBEDDING_MODEL=solar-embedding-1-large
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=jeju_travel
DEFAULT_NUM_RESULTS=5
MAX_TOKENS=1024
```

**또는 명령어로 한번에 생성:**
```bash
cat > .env << EOF
UPSTAGE_API_KEY=your_upstage_api_key_here
EMBEDDING_MODEL=solar-embedding-1-large
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=jeju_travel
DEFAULT_NUM_RESULTS=5
MAX_TOKENS=1024
EOF
```

### 4. 데이터베이스 초기화
```bash
python setup_database.py
```

## 💬 사용법

### 기본 사용법
```bash
python interactive_demo.py
```

### 컨텍스트 테스트 (대화 맥락 연결 테스트)
```bash
python interactive_demo.py --test-context
```

### 주요 명령어
- `quit` 또는 `exit`: 프로그램 종료
- `Ctrl+C`: 강제 종료

## 🎯 프롬프트 엔지니어링 가이드

### 📍 프롬프트 수정 위치

챗봇은 **두 가지 프롬프트**를 사용합니다:

1. **일정 생성 전용 프롬프트** (`plan_trip` intent)
2. **일반 추천 프롬프트** (기타 모든 intent)

#### 수정 파일 위치
- `interactive_demo.py` → `_generate_travel_recommendation_response()` 메서드
- `final_chatbot.py` → `_generate_travel_recommendation_response()` 메서드

### 🤖 현재 프롬프트 구조

#### 1. 일정 생성 프롬프트 (오르미 페르소나)
```python
if intent == "plan_trip":
    # 오르미 전용 프롬프트 사용
    # 특징: 단계별 CoT, 표 형태 출력, DB 검증 태그
```

#### 2. 일반 추천 프롬프트
```python
else:
    # 기존 친근한 대화체 프롬프트 사용
    # 특징: 자연스러운 대화, 개인화 추천
```

### ✨ 프롬프트 개선 포인트

#### 🎭 페르소나 개선
- **현재**: '오르미' 여행 추천 챗봇
- **개선 방향**: 더 구체적인 성격, 말투, 전문성 부여

#### 📝 응답 형식 개선
- **표 형태 일정**: 더 보기 좋은 형식으로 개선
- **DB 검증 표시**: "DB조회" 태그를 더 자연스럽게
- **단계별 설명**: 사용자에게 보여줄 부분과 내부 처리 분리

#### 🧠 추론 과정 개선
- **CoT (Chain of Thought)**: 내부 추론 과정 개선
- **정보 수집**: 더 효율적인 질문 방식
- **개인화**: 사용자 프로필 활용도 증대

### 🧪 테스트 시나리오

프롬프트 개선 후 다음 시나리오로 테스트:

#### 시나리오 1: 기본 일정 생성
```
사용자: "여자친구랑 3박4일 제주도 여행 일정 짜줘"
기대: 오르미 프롬프트 → 표 형태 일정 + DB조회 태그
```

#### 시나리오 2: 제약 조건 추가
```
사용자: "우유랑 치즈 못 먹어"
기대: 이전 대화 맥락 고려한 수정된 추천
```

#### 시나리오 3: 일관성 확인
```
사용자: "아까 3박4일이라고 했는데 왜 2박3일로 바뀌었어?"
기대: 맥락 인식하고 올바른 기간으로 수정
```

#### 시나리오 4: 일반 추천
```
사용자: "제주도 맛집 추천해줘"
기대: 일반 프롬프트 → 친근한 대화체 추천
```

### 📊 평가 기준

프롬프트 개선 후 다음 기준으로 평가:

1. **응답 품질** (1-5점)
   - 정보 정확성
   - 개인화 정도
   - 실용성

2. **대화 자연스러움** (1-5점)
   - 맥락 연결성
   - 말투 일관성
   - 사용자 경험

3. **기능 정확성** (1-5점)
   - 의도 분석 정확도
   - 프로필 정보 활용
   - 형식 준수 (표, 태그 등)

## 📁 프로젝트 구조

```
Prometheus-Team1/ (Prompt 브랜치)
├── README.md                 # 이 파일
├── requirements.txt          # 의존성 패키지  
├── .env.example             # 환경 변수 예시 (수동 생성 필요)
├── config.py                # 환경변수 기반 설정 파일
├── interactive_demo.py      # 🎯 메인 데모 파일 (프롬프트 수정 위치)
├── final_chatbot.py         # 🎯 챗봇 코어 로직 (프롬프트 수정 위치)  
├── setup_database.py        # 데이터베이스 초기화
├── data/                    # 제주도 여행 데이터
│   ├── visitjeju_event.json
│   ├── visitjeju_food.json
│   ├── visitjeju_hotel.json
│   └── visitjeju_tour.json
└── chroma_db/              # 벡터 데이터베이스 (자동 생성)
```

## 🔧 설정 방법

### 1. Upstage API 키 발급
1. [Upstage Console](https://console.upstage.ai/) 접속
2. 회원가입/로그인
3. API 키 발급
4. `.env` 파일에 추가

### 2. 데이터베이스 설정
```bash
# ChromaDB 초기화 및 제주도 데이터 로드
python setup_database.py
```

### 3. 모델 설정
- **LLM**: Solar Pro (Upstage)
- **Embedding**: solar-embedding-1-large
- **Vector DB**: ChromaDB

## 🚨 주의사항

### API 사용량 관리
- Solar Pro API는 유료 서비스입니다
- 테스트 시 API 사용량을 확인하세요
- 대량 테스트 전에 팀장에게 문의하세요

### 데이터 정확성
- 제주도 여행 정보는 2024년 기준입니다
- 실제 서비스 전 최신 정보로 업데이트가 필요할 수 있습니다

### 성능 최적화
- 첫 실행 시 데이터베이스 로딩으로 시간이 걸릴 수 있습니다
- 프롬프트가 길수록 API 응답 시간이 증가합니다

## 🆘 문제 해결

### 자주 발생하는 오류

#### 1. API 키 오류
```
openai.AuthenticationError: Invalid API key
```
**해결책**: `.env` 파일의 `UPSTAGE_API_KEY` 확인

#### 2. 데이터베이스 오류
```
chromadb.errors.InvalidDimensionException
```
**해결책**: `python setup_database.py` 재실행

#### 3. 의존성 오류
```
ModuleNotFoundError: No module named 'langchain_upstage'
```
**해결책**: `pip install -r requirements.txt` 재실행

### 로그 확인
실행 시 상세한 로그가 출력됩니다:
- `🔍 Solar Pro 쿼리 분석 중...`
- `👤 사용자 프로필 업데이트 중...`
- `🔎 여행 데이터 검색 중...`
- `💬 Solar Pro 응답 생성 중...`

문제 발생 시 해당 단계에서 멈추는지 확인하세요.

## 🔗 메인 프로젝트 연관성

### 📂 전체 오르다 프로젝트 구조
- **메인 브랜치**: Streamlit 기반 웹 애플리케이션 + Multi-agent RAG 시스템
- **Prompt 브랜치** (현재): 프롬프트 엔지니어링 전용 데모 환경

### 🔄 작업 플로우
1. **Prompt 브랜치**에서 프롬프트 개선 작업 수행
2. 테스트를 통해 검증된 프롬프트를 메인 브랜치에 반영
3. 메인 프로젝트의 Streamlit 앱에서 최종 테스트

### 📋 브랜치별 역할
| 브랜치 | 목적 | 주요 파일 |
|-------|------|----------|
| main | 전체 애플리케이션 | `main.py`, `src/` 폴더 전체 |
| **Prompt** | 프롬프트 튜닝 | `interactive_demo.py`, `final_chatbot.py` |

## 📞 지원

- **기술적 문제**: [GitHub Issues](https://github.com/DKUSeok2/Prometheus-Team1/issues)에 등록
- **프롬프트 아이디어**: 팀 채널에서 논의 후 Pull Request 생성
- **API 사용량**: 팀장 확인 후 사용 (Solar Pro API 유료)
- **메인 프로젝트**: [Prometheus Team1 Repository](https://github.com/DKUSeok2/Prometheus-Team1)

## 🤝 기여 방법

1. **Prompt 브랜치에서 작업**
   ```bash
   git checkout Prompt
   git pull origin Prompt
   ```

2. **프롬프트 개선 후 테스트**
   ```bash
   python interactive_demo.py --test-context
   ```

3. **검증 완료 시 Pull Request 생성**
   - Prompt → main 브랜치로 PR
   - 테스트 결과와 개선 사항 명시

---

**🌴 Happy Prompting! 🚀✨**

**팀원 모두가 함께 만드는 더 나은 제주도 여행 챗봇 '오르다'!** 