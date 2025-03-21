from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM

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
    
    # 'beomi/KoAlpaca-Polyglot-12.8B' 모델 사용 - 오픈 소스 한국어 모델
    model_id = "beomi/KoAlpaca-Polyglot-12.8B"
    
    # 또는 더 가벼운 대안으로:
    # model_id = "beomi/KoAlpaca-6.8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True
    )
    
    prompt = f"""
아래는 제주도 관광 정보입니다:
{context}

질문: {question}
답변:"""
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3,
        do_sample=True,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id
    )
    
    response = pipe(prompt)[0]["generated_text"]
    # 프롬프트 제거하고 답변만 반환
    answer = response.split("답변:")[1].strip()
    return answer

def main():
    # 벡터 DB 설정
    print("벡터 DB를 설정하는 중...")
    vectorstore = setup_vector_store()
    
    # 예시 질문들
    questions = [
        "한라산에 대해 알려주세요.",
        "우도에서 할 수 있는 활동은 무엇인가요?",
        "제주 올레길의 특징을 설명해주세요."
    ]
    
    # 각 질문에 대한 답변 생성
    for question in questions:
        print(f"\n질문: {question}")
        answer = get_answer(question, vectorstore)
        print(f"답변: {answer}")

if __name__ == "__main__":
    main() 