# 제주도 RAG 챗봇 예제

이 프로젝트는 제주도 관광지 정보를 벡터 DB에 저장하고, 사용자 질문에 대해 관련 정보를 검색한 후 답변을 생성하는 RAG(Retrieval-Augmented Generation) 시스템의 예제입니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

두 가지 버전의 예제 코드가 있습니다:

### 1. 기본 버전 (Mistral 7B 모델 사용)
```bash
python rag_example.py
```
- Mistral-7B-Instruct-v0.2 모델을 사용합니다
- GPU가 있으면 GPU를 활용합니다
- 메모리 사용량을 줄이기 위해 8비트 양자화를 사용합니다

### 2. 경량 버전
```bash
python rag_example_lightweight.py
```
- 더 가벼운 KETI-AIR/ke-t5-base-newslike 모델을 사용합니다
- CPU에서도 원활하게 실행됩니다
- 대화형 인터페이스로 제공됩니다

## 코드 구조

1. 임베딩 모델: sentence-transformers를 사용하여 텍스트를 벡터로 변환
2. 벡터 DB: Chroma DB를 사용하여 벡터 데이터 저장
3. 검색: 유사도 검색을 통해 관련 정보 검색
4. 생성: HuggingFace 모델을 사용하여 답변 생성

## 데이터

현재 예제에는 제주도의 주요 관광지 5곳에 대한 정보가 포함되어 있습니다:
- 한라산
- 성산일출봉
- 만장굴
- 우도
- 제주 올레길

더 많은 정보를 추가하여 시스템을 확장할 수 있습니다.

## 커스터마이징

- 다른 데이터로 변경: `jeju_data` 변수의 내용을 변경하거나, 파일에서 데이터를 읽어오도록 수정할 수 있습니다.
- 다른 임베딩 모델 사용: `HuggingFaceEmbeddings`의 `model_name` 파라미터를 변경하여 다른 임베딩 모델을 사용할 수 있습니다.
- 다른 생성 모델 사용: `model_id` 변수를 변경하여 다른 생성 모델을 사용할 수 있습니다. 