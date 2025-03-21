from typing import Annotated, Sequence, TypedDict, List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import os

# 환경 변수 로드
load_dotenv()

console = Console()

# 상태 타입 정의
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    context: str
    current_answer: str
    agent_outputs: List[str]
    final_answer: str

# 제주도 관광지 정보는 이전과 동일...
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
"""  # 이전 데이터와 동일

def show_agent_progress(agent_name: str, message: str):
    """에이전트 진행상황 표시"""
    console.print(Panel(
        f"[bold blue]{message}[/bold blue]",
        title=f"🤖 {agent_name}",
        border_style="blue"
    ))

def setup_vectorstore():
    """벡터 DB 설정 (이전과 동일)"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("벡터 DB 초기화 중...", total=None)
        # 이전 setup_vectorstore 코드와 동일
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(jeju_data)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory="./jeju_db"
        )
        return vectorstore

def search_agent(state: AgentState) -> AgentState:
    """검색 에이전트"""
    show_agent_progress("Search Agent 🔍", "관련 정보를 검색하고 있습니다...")
    
    vectorstore = setup_vectorstore()
    last_message = state["messages"][-1].content
    
    # k 값을 증가시키고 검색 결과 확인
    docs = vectorstore.similarity_search(last_message, k=3)
    context = "\n".join(doc.page_content for doc in docs)
    
    if not context.strip():
        context = jeju_data  # 검색 결과가 없으면 전체 데이터 사용
    
    show_agent_progress("Search Agent 🔍", "검색 완료!")
    
    return {
        **state,
        "context": context,
        "agent_outputs": state["agent_outputs"] + ["검색 결과 찾음"]
    }

def analysis_agent(state: AgentState) -> AgentState:
    """분석 에이전트"""
    show_agent_progress("Analysis Agent 📊", "검색된 정보를 분석하고 있습니다...")
    time.sleep(1)
    
    # 컨텍스트에서 핵심 정보 추출
    context = state["context"]
    question = state["messages"][-1].content
    
    # 간단한 키워드 기반 분석
    keywords = ["높이", "길이", "특징", "활동"]
    analyzed_info = []
    for keyword in keywords:
        if keyword in context:
            analyzed_info.append(f"{keyword} 정보 있음")
    
    show_agent_progress("Analysis Agent 📊", "분석 완료!")
    
    return {
        **state,
        "agent_outputs": state["agent_outputs"] + [f"분석된 정보: {', '.join(analyzed_info)}"]
    }

def generation_agent(state: AgentState) -> AgentState:
    """생성 에이전트"""
    show_agent_progress("Generation Agent 💭", "답변을 생성하고 있습니다...")
    
    if not state['context'].strip():
        return {
            **state,
            "current_answer": "검색 결과가 없습니다.",
            "agent_outputs": state["agent_outputs"] + ["답변 생성 실패"],
            "final_answer": "검색 결과가 없습니다."
        }
    
    # GPT-3.5-turbo 모델 초기화
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=500
    )
    
    # 프롬프트 템플릿 수정
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 제주도 관광 정보를 제공하는 도우미입니다.
주어진 정보를 바탕으로 질문에 답변해주세요.

답변 규칙:
1. 주어진 정보에 있는 내용만 사용하세요
2. 추측이나 일반적인 지식을 추가하지 마세요
3. 정보가 없다면 "주어진 정보에서 찾을 수 없습니다"라고 답변하세요
4. 한국어로 친절하게 답변해주세요"""),
        ("human", """다음은 제주도 관광 정보입니다:
{context}

질문: {question}

위 정보를 바탕으로 답변해주세요.""")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "context": state['context'],
            "question": state['messages'][-1].content
        })
        answer = response.content
    except Exception as e:
        print(f"\n[에러] 답변 생성 중 오류 발생: {str(e)}")
        answer = "답변 생성 중 오류가 발생했습니다."
    
    show_agent_progress("Generation Agent 💭", "답변 생성 완료!")
    
    return {
        **state,
        "current_answer": answer,
        "agent_outputs": state["agent_outputs"] + ["답변 생성됨"],
        "final_answer": answer
    }

def should_continue(state: AgentState) -> bool:
    """답변이 충분한지 확인"""
    # 항상 False를 반환하여 한 번의 사이클만 실행되도록 수정
    return False

def create_graph():
    """워크플로우 그래프 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("searcher", search_agent)
    workflow.add_node("analyzer", analysis_agent)
    workflow.add_node("generator", generation_agent)
    
    # 엣지 설정
    workflow.add_edge("searcher", "analyzer")
    workflow.add_edge("analyzer", "generator")
    workflow.add_conditional_edges(
        "generator",
        should_continue,
        {
            True: "searcher",
            False: END
        }
    )
    
    workflow.set_entry_point("searcher")
    
    return workflow.compile()

def main():
    console.print("[bold green]🤖 제주도 관광 정보 멀티에이전트 시스템 시작![/bold green]")
    console.print("=" * 50)
    
    chain = create_graph()
    
    while True:
        question = input("\n💬 질문을 입력하세요 (종료: q): ")
        if question.lower() == 'q':
            break
        
        console.print("\n[bold yellow]처리를 시작합니다...[/bold yellow]")
        console.print("=" * 50)
        
        state = {
            "messages": [HumanMessage(content=question)],
            "context": "",
            "current_answer": "",
            "agent_outputs": [],
            "final_answer": ""
        }
        
        result = chain.invoke(state)
        
        console.print("\n[bold green]최종 답변:[/bold green]")
        console.print(Panel(result["final_answer"], border_style="green"))
        console.print("=" * 50)

def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 시작 시간과 메모리 측정
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        # 함수 실행
        result = func(*args, **kwargs)
        
        # 종료 시간과 메모리 측정
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        # 결과 출력
        print(f"Function: {func.__name__}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        print(f"CPU usage: {end_cpu:.2f}%")
        
        return result
    return wrapper

if __name__ == "__main__":
    main() 