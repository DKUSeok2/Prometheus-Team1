# 🏗️ ORDA Backend API

**제주도 여행 AI 챗봇 '오르다'의 백엔드 API 서버**

> 이 브랜치는 [오르다 메인 프로젝트](https://github.com/DKUSeok2/Prometheus-Team1)의 백엔드 API 서버입니다.  
> FastAPI 기반으로 사용자 인증, 채팅 세션 관리, 챗봇 서비스 연동을 제공합니다.

## 📋 목차
- [🚀 빠른 시작](#-빠른-시작)
- [🛠️ 기술 스택](#️-기술-스택)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🔧 설치 및 실행](#-설치-및-실행)
- [🌐 API 엔드포인트](#-api-엔드포인트)
- [💾 데이터베이스](#-데이터베이스)

## 🚀 빠른 시작

```bash
# 1. 저장소 클론 (Backend 브랜치)
git clone -b Backend https://github.com/DKUSeok2/Prometheus-Team1.git
cd Prometheus-Team1

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 설정
cp config.example.py config.py
# config.py에서 데이터베이스 정보 및 JWT 키 설정

# 4. 데이터베이스 초기화 (필요 시)
python reset_db.py

# 5. 서버 실행
python main.py
```

## 🛠️ 기술 스택

- **Framework**: FastAPI
- **Database**: PostgreSQL + SQLAlchemy ORM
- **Authentication**: JWT (JSON Web Tokens)
- **Validation**: Pydantic
- **Server**: Uvicorn ASGI
- **HTTP Client**: HTTPX (챗봇 서비스 연동)

## 📁 프로젝트 구조

```
Backend/
├── README.md               # 이 파일
├── requirements.txt        # Python 의존성
├── config.example.py       # 설정 파일 예시
├── main.py                 # FastAPI 애플리케이션 진입점
├── models.py               # SQLAlchemy 데이터베이스 모델
├── schemas.py              # Pydantic 스키마 (입출력 검증)
├── database.py             # 데이터베이스 연결 설정
├── reset_db.py             # 데이터베이스 초기화 스크립트
├── routers/               # API 라우터
│   ├── auth.py            # 인증 관련 API
│   └── chat.py            # 채팅 관련 API
└── utils/                 # 유틸리티 모듈
    └── auth.py            # JWT 토큰 처리
```

## 🔧 설치 및 실행

### 1. 환경 준비
- Python 3.8 이상
- PostgreSQL 설치 및 실행
- 데이터베이스 생성 (`jeju_chatbot`)

### 2. 설정 파일 준비
```bash
cp config.example.py config.py
```

`config.py`에서 다음 항목을 수정하세요:
- `DATABASE_URL`: PostgreSQL 연결 정보
- `SECRET_KEY`: JWT 토큰 서명용 비밀키 (보안을 위해 복잡한 값으로 변경)

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 데이터베이스 초기화
```bash
# 기존 테이블 삭제 후 재생성
python reset_db.py
```

### 5. 서버 실행
```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 🌐 API 엔드포인트

### 📖 API 문서
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 🔐 인증 API (`/api/v1/auth`)
- `POST /register` - 사용자 회원가입
- `POST /login` - 사용자 로그인 (JWT 토큰 발급)
- `GET /me` - 현재 사용자 정보 조회

### 💬 채팅 API (`/api/v1/chat`)
- `POST /sessions` - 새 채팅 세션 생성
- `GET /sessions` - 사용자의 채팅 세션 목록
- `GET /sessions/{session_id}` - 특정 세션의 메시지 이력
- `POST /sessions/{session_id}/messages` - 채팅 메시지 송수신
- `PUT /sessions/{session_id}/title` - 세션 제목 변경
- `DELETE /sessions/{session_id}` - 세션 삭제

### 🤖 챗봇 연동
백엔드 API는 별도의 챗봇 서비스(`http://localhost:8001`)와 연동하여 AI 응답을 생성합니다.

## 💾 데이터베이스

### 🗃️ 주요 테이블
- **users**: 사용자 정보
- **chat_sessions**: 채팅 세션 (사용자별 대화 방)
- **chat_messages**: 채팅 메시지 (사용자/봇 메시지)

### 🔄 데이터베이스 관리
```bash
# 테이블 재생성
python reset_db.py

# 개발용 더미 데이터 생성 (필요 시)
# python create_demo_data.py
```

## 🔗 메인 프로젝트 연관성

### 📂 전체 오르다 프로젝트 구조
- **메인 브랜치**: Streamlit 기반 웹 애플리케이션
- **Prompt 브랜치**: 프롬프트 엔지니어링 데모
- **Backend 브랜치** (현재): FastAPI 백엔드 서버

### 🔄 서비스 연동 구조
```
Flutter App ↔ Backend API ↔ Chatbot Service ↔ ChromaDB + LLM
     ↓              ↓              ↓              ↓
  사용자 UI     인증/세션관리    AI 응답생성     벡터검색+생성
```

## 🚨 주의사항

### 보안 설정
- **SECRET_KEY**: 운영환경에서는 반드시 강력한 비밀키로 변경
- **DATABASE_URL**: 실제 데이터베이스 정보로 설정
- **.env 파일**: 민감한 정보는 환경 변수로 관리 권장

### 개발/운영 환경
- 개발: `http://localhost:8000`
- CORS 설정에 프론트엔드 도메인 추가 필요
- 운영환경에서는 HTTPS 사용 권장

## 🆘 문제 해결

### 자주 발생하는 오류

#### 1. 데이터베이스 연결 오류
```
sqlalchemy.exc.OperationalError: could not connect to server
```
**해결책**: PostgreSQL 서비스 실행 및 데이터베이스 생성 확인

#### 2. JWT 토큰 오류
```
401 Unauthorized: Could not validate credentials
```
**해결책**: 클라이언트에서 `Authorization: Bearer <token>` 헤더 포함

#### 3. 챗봇 서비스 연결 실패
```
챗봇 서비스 연결 실패: Connection refused
```
**해결책**: 챗봇 서비스(`http://localhost:8001`) 실행 상태 확인

## 📞 지원

- **기술적 문제**: [GitHub Issues](https://github.com/DKUSeok2/Prometheus-Team1/issues)
- **메인 프로젝트**: [Prometheus Team1 Repository](https://github.com/DKUSeok2/Prometheus-Team1)
- **API 문서**: http://localhost:8000/docs

---

**🏗️ 안정적이고 확장 가능한 백엔드로 오르다를 지원합니다!** 