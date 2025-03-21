from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# 제주도 관광지 정보
jeju_data = """
제주도는 한국의 가장 큰 섬으로, 아름다운 자연과 독특한 문화를 가지고 있습니다.

주요 관광지:
1. 한라산
- 높이: 1,947m
- 특징: 제주도의 상징적인 화산으로, 유네스코 세계자연유산으로 등재
- 활동: 등산, 야간 별보기

2. 성산일출봉
- 높이: 182m
- 특징: 화산 분출로 형성된 독특한 지형
- 활동: 일출 감상, 등산

3. 만장굴
- 길이: 약 7.4km
- 특징: 세계 최장의 용암동굴
- 활동: 동굴 탐험, 지질학 학습

4. 우도
- 특징: 소의 모양을 닮은 섬
- 활동: 자전거 여행, 해수욕

5. 제주 올레길
- 총 길이: 425km
- 특징: 제주도를 둘러싸는 걷기 여행길
- 활동: 트레킹, 자연 감상
"""

def setup_vector_store():
    # 텍스트 분할
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(jeju_data)
    
    # 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 벡터 DB 생성
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./jeju_db"
    )
    return vectorstore

def get_answer(question, vectorstore):
    # 검색 결과 가져오기
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # GPT-3.5-turbo 모델 초기화
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500
    )
    
    # 프롬프트 템플릿 수정
    prompt = f"""당신은 제주도 관광 정보를 제공하는 도우미입니다.
주어진 정보를 바탕으로 질문에 답변해주세요.

답변 규칙:
1. 주어진 정보에 있는 내용만 사용하세요
2. 추측이나 일반적인 지식을 추가하지 마세요
3. 정보가 없다면 "주어진 정보에서 찾을 수 없습니다"라고 답변하세요
4. 한국어로 친절하게 답변해주세요

주어진 정보:
{context}

질문: {question}

위 정보를 바탕으로 답변해주세요."""
    
    # 답변 생성
    response = llm.invoke(prompt)
    return response.content

def main():
    # 벡터 DB 설정
    print("벡터 DB를 설정하는 중...")
    vectorstore = setup_vector_store()
    
    # 대화형 인터페이스
    while True:
        question = input("\n질문을 입력하세요 (종료: q): ")
        if question.lower() == 'q':
            break
            
        answer = get_answer(question, vectorstore)
        print(f"\n답변: {answer}")

if __name__ == "__main__":
    main() 