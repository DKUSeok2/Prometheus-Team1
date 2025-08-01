# 🏔️ 오르다 (Orda) - Flutter 여행 앱

한국어 여행 동반자 애플리케이션

## 📱 앱 소개

오르다는 사용자의 여행을 함께하는 스마트한 여행 동반자 앱입니다.
- 🎯 개인화된 여행 추천
- 💬 AI 챗봇 여행 도우미  
- 🗺️ 카카오 지도 기반 위치 서비스
- 👤 사용자 맞춤 프로필 관리
- 🔐 안전한 로그인 시스템

## 🛠️ 기술 스택

- **Framework**: Flutter 3.32.6
- **Language**: Dart
- **Platform**: Web, iOS, Android, macOS, Linux, Windows
- **지도 서비스**: Kakao Map API
- **상태 관리**: SharedPreferences (로컬 저장소)
- **네트워킹**: HTTP 패키지
- **환경 변수**: flutter_dotenv

## 📦 설치 및 실행

### 1. 사전 요구사항

Flutter가 설치되어 있어야 합니다:

```bash
# Flutter 설치 (macOS - Homebrew)
brew install flutter

# Flutter 환경 확인
flutter doctor
```

### 2. 프로젝트 클론 및 의존성 설치

```bash
# 프로젝트 폴더로 이동
cd orda_app

# 의존성 설치
flutter pub get

# iOS 의존성 설치 (iOS에서 실행할 경우)
cd ios && pod install && cd ..
```

### 3. 환경 변수 설정 (.env 파일)

카카오 지도 API를 사용하기 위해 환경 변수 설정이 필요합니다:

```bash
# .env 파일 생성 (프로젝트 루트에)
touch .env
```

`.env` 파일에 다음 내용을 추가하세요:

```env
# 카카오 지도 API 키
# 카카오 개발자 사이트에서 발급받은 JavaScript API 키를 입력하세요
KAKAO_MAP_API_KEY=your_javascript_api_key_here
```

#### 🔑 카카오 지도 API 키 발급 방법
1. [카카오 개발자 사이트](https://developers.kakao.com) 접속
2. 카카오 계정으로 로그인
3. **내 애플리케이션 > 애플리케이션 추가하기**
4. 앱 이름 설정 (예: Orda Flutter App)
5. **플랫폼 설정 > Web 플랫폼 추가**
6. **앱 키 > JavaScript 키** 복사하여 `.env` 파일에 입력

### 4. 웹에서 실행 (추천)

#### 방법 1: 빌드 후 HTTP 서버 실행

```bash
# 1. 웹용 빌드
flutter build web

# 2. 빌드된 파일 디렉토리로 이동
cd build/web

# 3. HTTP 서버 실행
python3 -m http.server 8080

# 4. 브라우저에서 접속
# http://127.0.0.1:8080
```

#### 방법 2: Flutter 개발 서버 실행

```bash
# Flutter 개발 서버 실행
flutter run -d chrome

# 또는 포트 지정
flutter run -d web-server --web-port 3000
```

### 🛑 서버 종료 방법

#### HTTP 서버 종료 (방법 1 사용 시)
```bash
# 현재 실행 중인 Python HTTP 서버 확인
ps aux | grep "python3 -m http.server"

# 특정 포트(8080)를 사용하는 서버 종료
pkill -f "python3 -m http.server 8080"

# 또는 모든 Python HTTP 서버 종료
pkill -f "python3 -m http.server"

# 터미널에서 실행 중이라면 Ctrl+C
```

#### Flutter 개발 서버 종료 (방법 2 사용 시)
```bash
# 터미널에서 Ctrl+C 또는 q 입력

# 백그라운드에서 실행 중인 경우
ps aux | grep flutter
kill [프로세스ID]
```

### 5. 모바일에서 실행

#### Android
```bash
flutter run -d android
```

#### iOS (macOS에서만)
```bash
flutter run -d ios
```

## 🎮 앱 사용법

### 온보딩 플로우
1. **서비스 진입** - "진입하기" 버튼 클릭
2. **로그인 화면** - 이메일과 비밀번호로 로그인
3. **스플래시 화면** - "오르다" 로고 확인 (2초 자동 전환)
4. **온보딩 질문** - 3개 질문 페이지 완성
5. **메인 앱 진입** - "시작하기" 버튼으로 앱 시작

### 🔐 로그인 시스템
- **데모 계정**: 
  - 이메일: `demo@orda.com`
  - 비밀번호: `123456`
- **자동 로그인**: 로그인 상태가 로컬에 저장됩니다
- **로그아웃**: MY 화면에서 로그아웃 가능

### 주요 기능
- **🏠 홈**: 여행 상태와 캐릭터 확인
- **💬 채팅**: 오르다 봇과 실시간 대화
  - 채팅 기록 저장 및 관리
  - 사이드바에서 채팅 히스토리 확인
  - 채팅방 이름 변경 가능
  - 새로운 채팅방 생성 및 삭제
- **🗺️ 지도**: 카카오 지도 기반 위치 서비스
  - 현재 위치 표시
  - 마커 추가 및 관리
  - 실시간 지도 조작
- **👤 MY**: 프로필 관리 및 설정
  - 사용자 정보 표시
  - 로그아웃 기능

## 📁 프로젝트 구조

```
orda_app/
├── lib/
│   ├── main.dart                     # 앱 진입점 및 카카오 지도 초기화
│   ├── screens/                      # 화면 파일들
│   │   ├── onboarding_screen.dart    # 서비스 진입 화면
│   │   ├── login_screen.dart         # 로그인 화면
│   │   ├── splash_screen.dart        # 스플래시 화면
│   │   ├── onboarding_questions_screen.dart  # 온보딩 질문
│   │   ├── home_screen.dart          # 메인 홈 화면
│   │   ├── chat_screen.dart          # 채팅 화면 (사이드바 포함)
│   │   ├── map_screen.dart           # 카카오 지도 화면
│   │   └── my_screen.dart            # MY 화면
│   ├── services/                     # 서비스 로직
│   │   ├── auth_service.dart         # 인증 서비스
│   │   └── chat_service.dart         # 채팅 서비스
│   ├── widgets/                      # 공통 위젯
│   ├── models/                       # 데이터 모델
│   └── utils/                        # 유틸리티
├── build/web/                        # 웹 빌드 결과물
├── .env                              # 환경 변수 (API 키 등)
├── pubspec.yaml                      # 의존성 관리
└── README.md                         # 이 파일
```

## 📦 주요 의존성

```yaml
dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.8
  http: ^1.1.0                    # HTTP 통신
  shared_preferences: ^2.2.2      # 로컬 저장소
  email_validator: ^2.1.17        # 이메일 검증
  kakao_map_plugin: ^0.3.5        # 카카오 지도
  flutter_dotenv: ^5.1.0          # 환경 변수 관리
```

## 🎨 UI/UX 특징

- **한국어 최적화**: 한국 사용자를 위한 UI/UX
- **반응형 디자인**: 다양한 화면 크기 지원
- **직관적 네비게이션**: 하단 탭 바 구조
- **부드러운 애니메이션**: 자연스러운 화면 전환
- **실시간 채팅**: 사이드바와 함께 제공되는 채팅 기능
- **인터랙티브 지도**: 카카오 지도 기반 위치 서비스

## 🔧 개발 환경

```bash
# 코드 분석
flutter analyze

# 테스트 실행
flutter test

# 빌드 (각 플랫폼별)
flutter build web      # 웹
flutter build apk       # Android APK
flutter build ios       # iOS (macOS에서만)
```

## 🐛 문제 해결

### 웹 실행 시 빈 화면이 나올 때
```bash
# 캐시 정리 후 재빌드
flutter clean
flutter pub get
flutter build web

# 브라우저 캐시 비우기 (Ctrl+Shift+R 또는 Cmd+Shift+R)
```

### 의존성 문제 발생 시
```bash
# 의존성 업그레이드
flutter pub upgrade

# pubspec.lock 삭제 후 재설치
rm pubspec.lock
flutter pub get
```

### iOS 빌드 문제 발생 시
```bash
# iOS 의존성 재설치
cd ios
pod deintegrate
pod install
cd ..

# 또는 캐시 정리
cd ios
pod cache clean --all
pod install
cd ..
```

### 카카오 지도 관련 문제
```bash
# .env 파일이 제대로 설정되었는지 확인
cat .env

# API 키가 올바른지 확인
# 카카오 개발자 사이트에서 JavaScript 키를 사용했는지 확인
```

### 포트 충돌 문제 발생 시
```bash
# 8080 포트를 사용하는 프로세스 확인
lsof -i :8080

# 해당 프로세스 종료
kill [프로세스ID]

# 또는 Python HTTP 서버만 종료
pkill -f "python3 -m http.server"

# 다른 포트 사용
python3 -m http.server 3000  # 3000번 포트로 실행
```

## 📝 개발 정보

- **개발자**: 오르다 팀
- **버전**: 1.0.0
- **라이선스**: MIT
- **개발 시작일**: 2025년 1월

## 🚀 향후 계획

- [x] 로그인 시스템 구현
- [x] 카카오 지도 API 연동
- [x] 채팅 기능 및 히스토리 관리
- [x] 환경 변수 관리 (.env)
- [ ] AI 챗봇 고도화
- [ ] 여행 계획 기능 추가
- [ ] 소셜 기능 구현
- [ ] 오프라인 지원
- [ ] 푸시 알림 기능

---

## 💡 팁

### 🌐 웹 실행 팁
- 웹에서 실행할 때는 **http://127.0.0.1:8080**으로 정확히 접속하세요
- 개발 중에는 `flutter run -d chrome`이 가장 편리합니다
- 브라우저 캐시 문제가 있을 때는 **Ctrl+Shift+R** (하드 리프레시)

### 🔑 API 키 관리 팁
- `.env` 파일은 Git에 커밋되지 않으므로 안전합니다
- 팀원들과 공유할 때는 `.env.example` 파일을 만들어 참고하게 하세요
- 카카오 개발자 사이트에서 **JavaScript 키**를 사용해야 합니다 (Native 키 아님)

### 🗺️ 지도 사용 팁
- 카카오 지도는 인터넷 연결이 필요합니다
- 위치 권한이 필요할 수 있으니 브라우저에서 허용해주세요
- 지도 로딩이 느릴 때는 API 키를 확인해보세요

### 💬 채팅 기능 팁
- 채팅 기록은 로컬에 저장되어 앱을 다시 실행해도 유지됩니다
- 사이드바에서 채팅방 이름을 클릭하면 편집할 수 있습니다
- 새로운 채팅방 생성 시 자동으로 고유한 ID가 부여됩니다

### 🔧 서버 관리 팁
- 서버 종료 후에는 포트가 완전히 해제될 때까지 잠시 기다리세요
- 여러 포트(8080, 3000, 8000 등)를 번갈아 사용하면 편리합니다
- `lsof -i :8080` 명령어로 포트 사용 상태를 확인할 수 있습니다

### 🛠️ 개발 팁
- 문제 발생 시 `flutter doctor`로 환경을 먼저 확인하세요
- 코드 변경 후에는 `flutter clean && flutter build web`로 깨끗하게 재빌드
- VS Code나 Android Studio의 Flutter 확장을 사용하면 더 편리합니다

**즐거운 여행 되세요! 🎒✈️**
