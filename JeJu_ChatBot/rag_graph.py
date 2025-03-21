from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.graphs import Graph
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import operator
from typing import List, Tuple
from langgraph.graph import END, StateGraph

# 상태 타입 정의
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    context: str
    current_answer: str

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

def setup_vectorstore():
    # 텍스트 분할
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(jeju_data)
    
    # 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 벡터 DB 생성
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./jeju_db"
    )
    return vectorstore

def retrieve(state: AgentState) -> AgentState:
    """벡터 DB에서 관련 정보 검색"""
    vectorstore = setup_vectorstore()
    
    # 마지막 사용자 메시지 가져오기
    last_message = state["messages"][-1].content
    
    # 관련 문서 검색
    docs = vectorstore.similarity_search(last_message, k=2)
    context = "\n".join(doc.page_content for doc in docs)
    
    return {
        **state,
        "context": context
    }

def generate_answer(state: AgentState) -> AgentState:
    """검색된 정보를 바탕으로 답변 생성"""
    # 모델 설정
    model_id = "beomi/KoAlpaca-6.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3
    )
    
    # 프롬프트 생성
    prompt = f"""
다음 정보를 바탕으로 질문에 답변해주세요:

정보:
{state['context']}

질문: {state['messages'][-1].content}

답변:"""
    
    # 답변 생성
    response = pipe(prompt)[0]["generated_text"]
    answer = response.split("답변:")[-1].strip()
    
    return {
        **state,
        "current_answer": answer
    }

def should_continue(state: AgentState) -> bool:
    """답변이 충분한지 확인"""
    return len(state["current_answer"]) < 50

def create_graph():
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("retriever", retrieve)
    workflow.add_node("generator", generate_answer)
    
    # 엣지 설정
    workflow.add_edge("retriever", "generator")
    workflow.add_conditional_edges(
        "generator",
        should_continue,
        {
            True: "retriever",
            False: END
        }
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("retriever")
    
    return workflow.compile()

def main():
    # 그래프 생성
    chain = create_graph()
    
    # 대화형 인터페이스
    while True:
        question = input("\n질문을 입력하세요 (종료: q): ")
        if question.lower() == 'q':
            break
        
        # 초기 상태 설정
        state = {
            "messages": [HumanMessage(content=question)],
            "context": "",
            "current_answer": ""
        }
        
        # 실행
        result = chain.invoke(state)
        print(f"\n답변: {result['current_answer']}")

if __name__ == "__main__":
    main() 