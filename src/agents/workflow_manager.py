"""
LangGraph 기반 워크플로우 매니저
"""
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from .query_analyzer import QueryAnalyzer
from .search_agents import CATEGORY_AGENTS
from .response_generator import ResponseGenerator
from .flight_agent import flight_agent

class WorkflowState(TypedDict):
    """워크플로우 상태 정의"""
    query: str
    chat_history: List[Dict]
    analysis: Dict[str, Any]
    search_results: Dict[str, List]
    flight_results: Dict[str, Any]
    final_response: str
    error: str

class WorkflowManager:
    """Multi-agent 워크플로우 관리자"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.response_generator = ResponseGenerator()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        
        workflow = StateGraph(WorkflowState)
        
        # 노드 추가
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("search_data", self._search_data_node)
        workflow.add_node("search_flights", self._search_flights_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # 엣지 설정
        workflow.set_entry_point("analyze_query")
        
        # 쿼리 분석 후 검색 타입 결정
        workflow.add_conditional_edges(
            "analyze_query",
            self._determine_search_type,
            {
                "travel_search": "search_data",
                "flight_search": "search_flights",
                "error": "handle_error",
                "end": END
            }
        )
        
        # 검색 후 응답 생성
        workflow.add_edge("search_data", "generate_response")
        workflow.add_edge("search_flights", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _analyze_query_node(self, state: WorkflowState) -> WorkflowState:
        """쿼리 분석 노드"""
        try:
            result = self.query_analyzer.process(
                query=state["query"],
                context={"chat_history": state.get("chat_history", [])}
            )
            
            if "error" in result["metadata"]:
                state["error"] = result["metadata"]["error"]
            else:
                state["analysis"] = result["metadata"]["analysis"]
            
            return state
            
        except Exception as e:
            state["error"] = f"쿼리 분석 중 오류: {str(e)}"
            return state
    
    def _search_data_node(self, state: WorkflowState) -> WorkflowState:
        """데이터 검색 노드"""
        try:
            analysis = state.get("analysis", {})
            search_results = {}
            
            # 검색할 카테고리 결정
            categories = self.query_analyzer.get_search_categories(analysis)
            
            # 각 카테고리별 검색 수행
            for category in categories:
                if category in CATEGORY_AGENTS:
                    agent = CATEGORY_AGENTS[category]
                    result = agent.process(
                        query=state["query"],
                        context={"max_results": 3}
                    )
                    
                    if "error" not in result["metadata"]:
                        search_results[category] = result["metadata"].get("results", [])
                    else:
                        print(f"카테고리 {category} 검색 오류: {result['metadata']['error']}")
            
            state["search_results"] = search_results
            return state
            
        except Exception as e:
            state["error"] = f"데이터 검색 중 오류: {str(e)}"
            return state
    
    def _generate_response_node(self, state: WorkflowState) -> WorkflowState:
        """응답 생성 노드"""
        try:
            # 항공권 검색 결과가 있는 경우 먼저 처리
            if state.get("flight_results"):
                flight_summary = flight_agent.generate_flight_summary(state["flight_results"])
                state["final_response"] = flight_summary
                return state
            
            # 일반 여행 정보 응답 생성
            result = self.response_generator.process(
                query=state["query"],
                context={
                    "search_results": state.get("search_results", {}),
                    "analysis": state.get("analysis", {}),
                    "chat_history": state.get("chat_history", [])
                }
            )
            
            if "error" in result["metadata"]:
                state["error"] = result["metadata"]["error"]
            else:
                state["final_response"] = result["content"]
            
            return state
            
        except Exception as e:
            state["error"] = f"응답 생성 중 오류: {str(e)}"
            return state
    
    def _search_flights_node(self, state: WorkflowState) -> WorkflowState:
        """항공권 검색 노드"""
        try:
            query = state["query"]
            
            # 항공권 관련 쿼리인지 확인
            if not flight_agent.is_flight_related_query(query):
                state["error"] = "항공권 관련 정보를 찾을 수 없습니다."
                return state
            
            # 쿼리에서 항공편 정보 추출
            flight_info = flight_agent.extract_flight_info(query)
            
            # 항공편 검색 수행
            flight_results = flight_agent.search_flights(flight_info)
            
            state["flight_results"] = flight_results
            return state
            
        except Exception as e:
            state["error"] = f"항공권 검색 중 오류: {str(e)}"
            return state
    
    def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """에러 처리 노드"""
        error_message = state.get("error", "알 수 없는 오류가 발생했습니다.")
        state["final_response"] = f"죄송합니다. {error_message} 다시 시도해주세요."
        return state
    
    def _determine_search_type(self, state: WorkflowState) -> str:
        """검색 타입 결정"""
        if state.get("error"):
            return "error"
        
        analysis = state.get("analysis", {})
        intent = analysis.get("intent", "")
        categories = analysis.get("categories", [])
        
        # 일반 대화인 경우 검색하지 않음
        if intent == "일반대화":
            state["final_response"] = "안녕하세요! 제주도 여행에 대해 궁금한 것이 있으시면 언제든 물어보세요. 맛집, 숙소, 관광지, 행사, 항공편 정보를 도와드릴 수 있습니다."
            return "end"
        
        # 항공권 검색인지 확인
        if intent == "항공권검색" or "항공" in categories:
            return "flight_search"
        
        # 일반 여행 정보 검색
        return "travel_search"
    
    def process_query(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """쿼리 처리 메인 함수"""
        
        # 초기 상태 설정
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "analysis": {},
            "search_results": {},
            "flight_results": {},
            "final_response": "",
            "error": ""
        }
        
        try:
            # 워크플로우 실행
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "response": final_state.get("final_response", "응답을 생성할 수 없습니다."),
                "analysis": final_state.get("analysis", {}),
                "search_results": final_state.get("search_results", {}),
                "flight_results": final_state.get("flight_results", {}),
                "error": final_state.get("error"),
                "success": not bool(final_state.get("error"))
            }
            
        except Exception as e:
            return {
                "response": f"처리 중 오류가 발생했습니다: {str(e)}",
                "analysis": {},
                "search_results": {},
                "flight_results": {},
                "error": str(e),
                "success": False
            }

# 전역 워크플로우 매니저 인스턴스
workflow_manager = WorkflowManager()

