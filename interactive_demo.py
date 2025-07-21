"""
제주도 여행 챗봇 인터랙티브 데모
- 사용자 쿼리 분석 후 필요한 정보 질문
- 프로필 점진적 구축
- 실시간 대화 모드
"""

import chromadb
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# LangGraph 
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Upstage APIs
from langchain_upstage import UpstageEmbeddings, ChatUpstage

# Configuration
from config import (
    UPSTAGE_API_KEY, EMBEDDING_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME, DEFAULT_NUM_RESULTS
)

@dataclass
class UserProfile:
    """사용자 프로필 정보"""
    travel_companions: Optional[str] = None  # 동행자
    duration: Optional[str] = None          # 여행 기간
    interests: List[str] = None             # 관심사
    age_group: Optional[str] = None         # 연령대
    budget: Optional[str] = None            # 예산
    accommodation_type: Optional[str] = None # 숙박 유형
    transportation: Optional[str] = None     # 교통수단
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
    
    def get_missing_info(self) -> List[str]:
        """부족한 정보 항목 반환"""
        missing = []
        if not self.travel_companions:
            missing.append("동행자")
        if not self.duration:
            missing.append("여행기간")
        if not self.interests:
            missing.append("관심사")
        if not self.budget:
            missing.append("예산")
        return missing
    
    def completion_rate(self) -> float:
        """프로필 완성도 (0-1)"""
        total_fields = 6  # 주요 필드 수
        completed = sum([
            bool(self.travel_companions),
            bool(self.duration), 
            bool(self.interests),
            bool(self.budget),
            bool(self.accommodation_type),
            bool(self.transportation)
        ])
        return completed / total_fields
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

@dataclass
class ChatbotState:
    """챗봇 상태 정보"""
    user_message: str
    conversation_history: List[Dict] = None
    user_profile: UserProfile = None
    search_results: List[Dict] = None
    analysis_result: Dict = None
    response: str = ""
    needs_more_info: bool = False
    follow_up_questions: List[str] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_profile is None:
            self.user_profile = UserProfile()
        if self.follow_up_questions is None:
            self.follow_up_questions = []

class InteractiveJejuChatbot:
    """인터랙티브 제주도 여행 챗봇"""
    
    def __init__(self):
        """챗봇 초기화"""
        print("🌟 인터랙티브 제주도 여행 챗봇 초기화 중...")
        
        # Upstage 임베딩 모델
        self.embeddings = UpstageEmbeddings(
            api_key=UPSTAGE_API_KEY,
            model=EMBEDDING_MODEL
        )
        print(f"✅ Upstage 임베딩 모델 로드: {EMBEDDING_MODEL}")
        
        # Upstage Solar Pro API
        self.chat_model = ChatUpstage(
            api_key=UPSTAGE_API_KEY,
            model="solar-pro",
            temperature=0.7
        )
        print("✅ Upstage Solar Pro API 연결 성공")
        
        # ChromaDB 클라이언트
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB 컬렉션 연결: {COLLECTION_NAME}")
        
        # 메모리 체크포인트
        self.memory = MemorySaver()
        
        # 멀티 에이전트 그래프 구성
        self.graph = self._build_agent_graph()
        print("✅ 인터랙티브 멀티 에이전트 그래프 구성 완료")
        
    def _build_agent_graph(self) -> StateGraph:
        """멀티 에이전트 그래프 구성"""
        
        graph = StateGraph(ChatbotState)
        
        # 에이전트 노드 추가
        graph.add_node("query_analyzer", self.analyze_query_with_solar_pro)
        graph.add_node("profile_updater", self.update_user_profile)  
        graph.add_node("info_checker", self.check_missing_info)
        graph.add_node("search_agent", self.search_travel_data)
        graph.add_node("response_generator", self.generate_response_with_solar_pro)
        
        # 조건부 엣지 (정보가 부족하면 질문, 충분하면 검색)
        graph.add_edge("query_analyzer", "profile_updater")
        graph.add_edge("profile_updater", "info_checker")
        
        def should_ask_more_info(state: ChatbotState) -> str:
            """정보가 부족한지 판단"""
            if state.needs_more_info:
                return "response_generator"  # 질문 생성
            else:
                return "search_agent"  # 검색 진행
        
        graph.add_conditional_edges(
            "info_checker",
            should_ask_more_info,
            {
                "response_generator": "response_generator",
                "search_agent": "search_agent"
            }
        )
        
        graph.add_edge("search_agent", "response_generator")
        graph.add_edge("response_generator", END)
        
        # 시작점 설정
        graph.set_entry_point("query_analyzer")
        
        return graph.compile(checkpointer=self.memory)
    
    def analyze_query_with_solar_pro(self, state: ChatbotState) -> ChatbotState:
        """1단계: Solar Pro API로 쿼리 분석"""
        print("🔍 Solar Pro 쿼리 분석 중...")
        
        try:
            analysis_prompt = f"""제주도 여행 전문 상담사로서 사용자의 질문을 정확하게 분석해주세요.

사용자 메시지: "{state.user_message}"

다음 JSON 형식으로 분석 결과를 제공해주세요:
{{
    "intent": "find_food|find_hotel|find_attraction|plan_trip|greeting|general_info|provide_info 중 하나",
    "companions": "가족|친구|연인|혼자|불명 중 하나", 
    "duration": "당일|1박2일|2박3일|3박4일|4박5일|장기|불명 중 하나",
    "budget": "절약|보통|여유|럭셔리|불명 중 하나",
    "interests": ["음식", "바다", "산", "숙박", "관광지", "문화", "액티비티", "쇼핑", "휴양", "사진"] 중 해당하는 것들,
    "transportation": "렌터카|대중교통|택시|도보|불명 중 하나",
    "accommodation_preference": "호텔|리조트|펜션|게스트하우스|민박|불명 중 하나",
    "confidence": 0.0~1.0 사이의 확신도,
    "user_providing_info": true/false (사용자가 정보를 제공하고 있는지),
    "reasoning": "분석 근거"
}}

사용자의 메시지를 세심하게 분석하여 위 형식으로만 답변해주세요."""

            # Solar Pro API로 분석
            response = self.chat_model.invoke(analysis_prompt)
            analysis_response = response.content.strip()
            
            print(f"   🌟 Solar Pro 분석: {analysis_response[:150]}...")
            
            # JSON 파싱
            try:
                if '{' in analysis_response and '}' in analysis_response:
                    json_start = analysis_response.find('{')
                    json_end = analysis_response.rfind('}') + 1
                    json_str = analysis_response[json_start:json_end]
                    parsed_analysis = json.loads(json_str)
                    
                    intent = parsed_analysis.get("intent", "general_info")
                    travel_info = {
                        "companions": parsed_analysis.get("companions"),
                        "duration": parsed_analysis.get("duration"),
                        "budget": parsed_analysis.get("budget"),
                        "interests": parsed_analysis.get("interests", []),
                        "transportation": parsed_analysis.get("transportation"),
                        "accommodation_preference": parsed_analysis.get("accommodation_preference")
                    }
                    confidence = parsed_analysis.get("confidence", 0.8)
                    user_providing_info = parsed_analysis.get("user_providing_info", False)
                    reasoning = parsed_analysis.get("reasoning", "Solar Pro 분석")
                    
                else:
                    raise ValueError("JSON 형식을 찾을 수 없음")
                    
            except Exception as parse_error:
                print(f"   ⚠️ JSON 파싱 실패, 키워드 기반 분석: {parse_error}")
                
                # 백업 분석
                intent = self._classify_intent_by_keywords(state.user_message, analysis_response)
                travel_info = self._extract_info_by_keywords(state.user_message, analysis_response)
                confidence = 0.7
                user_providing_info = any(word in state.user_message.lower() for word in 
                                        ["명", "박", "일", "가족", "친구", "혼자", "만원", "원"])
                reasoning = "키워드 기반 분석"
            
            state.analysis_result = {
                "intent": intent,
                "travel_info": travel_info,
                "confidence": confidence,
                "user_providing_info": user_providing_info,
                "reasoning": reasoning,
                "solar_analysis": analysis_response
            }
            
            print(f"   ✅ 분석 결과: {intent} (확신도: {confidence:.2f})")
            
        except Exception as e:
            print(f"   ❌ 쿼리 분석 실패: {e}")
            state.analysis_result = {
                "intent": "general_info",
                "travel_info": {},
                "confidence": 0.5,
                "user_providing_info": False,
                "reasoning": "분석 실패"
            }
        
        return state
    
    def _classify_intent_by_keywords(self, user_message: str, analysis: str) -> str:
        """키워드 기반 의도 분류"""
        text = (user_message + " " + analysis).lower()
        
        if any(word in text for word in ["음식", "맛집", "식당", "먹거리"]):
            return "find_food"
        elif any(word in text for word in ["숙박", "호텔", "민박", "펜션"]):
            return "find_hotel"
        elif any(word in text for word in ["관광", "명소", "여행지", "볼거리"]):
            return "find_attraction"
        elif any(word in text for word in ["일정", "계획", "여행"]):
            return "plan_trip"
        elif any(word in user_message.lower() for word in ["안녕", "hello", "hi"]):
            return "greeting"
        else:
            return "general_info"
    
    def _extract_info_by_keywords(self, user_message: str, analysis: str) -> dict:
        """키워드 기반 정보 추출"""
        text = (user_message + " " + analysis).lower()
        
        info = {}
        
        # 동행자
        if "가족" in text:
            info["companions"] = "가족"
        elif "친구" in text:
            info["companions"] = "친구"
        elif any(word in text for word in ["연인", "아내", "남편", "와이프", "남친", "여친"]):
            info["companions"] = "연인"
        elif "혼자" in text:
            info["companions"] = "혼자"
        
        # 기간
        if any(period in text for period in ["3박4일", "3박"]):
            info["duration"] = "3박4일"
        elif any(period in text for period in ["2박3일", "2박"]):
            info["duration"] = "2박3일"
        elif any(period in text for period in ["1박2일", "1박"]):
            info["duration"] = "1박2일"
        elif "당일" in text:
            info["duration"] = "당일"
        
        # 관심사
        interests = []
        if any(word in text for word in ["음식", "맛집", "먹거리"]):
            interests.append("음식")
        if any(word in text for word in ["바다", "해변", "해수욕장"]):
            interests.append("바다")
        if any(word in text for word in ["산", "한라산", "등산", "오름"]):
            interests.append("산")
        if any(word in text for word in ["문화", "역사", "박물관"]):
            interests.append("문화")
        if any(word in text for word in ["액티비티", "체험", "스포츠"]):
            interests.append("액티비티")
            
        info["interests"] = interests
        
        return info
    
    def update_user_profile(self, state: ChatbotState) -> ChatbotState:
        """2단계: 사용자 프로필 업데이트"""
        print("👤 사용자 프로필 업데이트 중...")
        
        if state.analysis_result:
            travel_info = state.analysis_result.get("travel_info", {})
            
            # 프로필 정보 업데이트 (기존 정보 유지)
            if travel_info.get("companions") and travel_info["companions"] != "불명":
                state.user_profile.travel_companions = travel_info["companions"]
            if travel_info.get("duration") and travel_info["duration"] != "불명":
                state.user_profile.duration = travel_info["duration"]
            if travel_info.get("budget") and travel_info["budget"] != "불명":
                state.user_profile.budget = travel_info["budget"]
            if travel_info.get("transportation") and travel_info["transportation"] != "불명":
                state.user_profile.transportation = travel_info["transportation"]
            if travel_info.get("accommodation_preference") and travel_info["accommodation_preference"] != "불명":
                state.user_profile.accommodation_type = travel_info["accommodation_preference"]
            
            # 관심사 추가 (중복 제거)
            if travel_info.get("interests"):
                for interest in travel_info["interests"]:
                    if interest not in state.user_profile.interests:
                        state.user_profile.interests.append(interest)
        
        completion = state.user_profile.completion_rate()
        print(f"   ✅ 프로필 완성도: {completion:.1%}")
        print(f"   👥 동행자: {state.user_profile.travel_companions}")
        print(f"   📅 기간: {state.user_profile.duration}")
        print(f"   💰 예산: {state.user_profile.budget}")
        print(f"   ❤️ 관심사: {state.user_profile.interests}")
        
        return state
    
    def check_missing_info(self, state: ChatbotState) -> ChatbotState:
        """3단계: 필요 정보 부족 여부 확인"""
        print("🔍 필요 정보 확인 중...")
        
        missing_info = state.user_profile.get_missing_info()
        intent = state.analysis_result.get("intent", "general_info")
        
        # 구체적인 요청이 있을 때는 정보 수집 우선
        needs_specific_info = intent in ["plan_trip", "find_food", "find_hotel", "find_attraction"]
        completion_rate = state.user_profile.completion_rate()
        
        # 정보 수집이 필요한 조건
        state.needs_more_info = (
            needs_specific_info and 
            completion_rate < 0.7 and  # 70% 미만 완성도
            len(missing_info) > 0 and
            intent != "greeting"
        )
        
        if state.needs_more_info:
            # 우선순위에 따른 질문 생성
            priority_questions = self._generate_priority_questions(missing_info, intent)
            state.follow_up_questions = priority_questions[:2]  # 최대 2개 질문
            print(f"   ⚠️ 추가 정보 필요: {missing_info}")
        else:
            print(f"   ✅ 정보 충분 (완성도: {completion_rate:.1%})")
        
        return state
    
    def _generate_priority_questions(self, missing_info: List[str], intent: str) -> List[str]:
        """우선순위 기반 질문 생성"""
        questions = []
        
        # 의도별 우선순위 정보
        priority_map = {
            "plan_trip": ["동행자", "여행기간", "예산", "관심사"],
            "find_food": ["관심사", "동행자", "예산"],
            "find_hotel": ["동행자", "여행기간", "예산"],
            "find_attraction": ["관심사", "동행자", "여행기간"]
        }
        
        priority_info = priority_map.get(intent, ["동행자", "여행기간"])
        
        # 우선순위 순서로 질문 생성
        for info_type in priority_info:
            if info_type in missing_info:
                if info_type == "동행자":
                    questions.append("누구와 함께 여행을 가시나요? (가족/친구/연인/혼자)")
                elif info_type == "여행기간":
                    questions.append("몇 박 며칠 일정으로 계획하고 계시나요?")
                elif info_type == "예산":
                    questions.append("여행 예산은 어느 정도로 생각하고 계시나요?")
                elif info_type == "관심사":
                    questions.append("어떤 것에 가장 관심이 있으신가요? (음식/바다/산/문화/액티비티 등)")
        
        return questions
    
    def search_travel_data(self, state: ChatbotState) -> ChatbotState:
        """4단계: 여행 데이터 검색"""
        print("🔎 여행 데이터 검색 중...")
        
        try:
            # 검색 쿼리 생성
            search_query = state.user_message
            
            # 사용자 프로필 기반 쿼리 강화
            if state.user_profile.interests:
                search_query += " " + " ".join(state.user_profile.interests)
            
            if state.user_profile.budget:
                search_query += f" {state.user_profile.budget}"
            
            # ChromaDB 검색
            results = self.collection.query(
                query_texts=[search_query],
                n_results=DEFAULT_NUM_RESULTS,
                include=['documents', 'metadatas']
            )
            
            # 검색 결과 정리
            search_results = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                search_results.append({
                    'content': doc,
                    'metadata': metadata
                })
            
            state.search_results = search_results
            print(f"   ✅ 검색 결과: {len(search_results)}개")
            
        except Exception as e:
            print(f"   ❌ 검색 실패: {e}")
            state.search_results = []
        
        return state
    
    def generate_response_with_solar_pro(self, state: ChatbotState) -> ChatbotState:
        """5단계: Solar Pro 응답 생성"""
        print("💬 Solar Pro 응답 생성 중...")
        
        try:
            if state.needs_more_info:
                # 정보 수집용 응답
                response = self._generate_info_collection_response(state)
            else:
                # 정보 제공용 응답
                response = self._generate_travel_recommendation_response(state)
            
            state.response = response
            print("   ✅ Solar Pro 응답 생성 완료")
            
        except Exception as e:
            print(f"   ❌ 응답 생성 실패: {e}")
            state.response = "죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요. 🙏"
        
        # 대화 이력 업데이트
        state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": state.user_message,
            "bot_response": state.response,
            "profile_completion": state.user_profile.completion_rate(),
            "needs_more_info": state.needs_more_info
        })
        
        return state
    
    def _generate_info_collection_response(self, state: ChatbotState) -> str:
        """정보 수집을 위한 응답 생성"""
        
        # 현재 파악된 정보 요약
        profile_summary = ""
        if state.user_profile.travel_companions:
            profile_summary += f"👥 {state.user_profile.travel_companions}과 함께, "
        if state.user_profile.duration:
            profile_summary += f"📅 {state.user_profile.duration} 일정, "
        if state.user_profile.interests:
            profile_summary += f"❤️ {', '.join(state.user_profile.interests)} 관심 "
        
        # 대화 히스토리 컨텍스트 추가
        conversation_context = ""
        if state.conversation_history and len(state.conversation_history) > 0:
            recent_messages = state.conversation_history[-3:]  # 최근 3개 메시지만
            conversation_context = "\n이전 대화:\n"
            for msg in recent_messages:
                role = "사용자" if msg["role"] == "user" else "상담사"
                conversation_context += f"- {role}: {msg['content'][:100]}...\n"
        
        # 질문 생성 프롬프트
        question_prompt = f"""제주도 여행 전문 상담사로서 사용자의 여행 계획을 도와주세요.

현재까지 파악된 정보: {profile_summary}

{conversation_context}

사용자 메시지: "{state.user_message}"

추가로 알아야 할 정보: {state.follow_up_questions}

다음 가이드라인으로 응답해주세요:
1. 친근하고 자연스러운 톤으로 답변
2. 이전 대화의 맥락을 고려한 연속적인 대화
3. 현재까지 파악된 정보에 대해 간단히 확인
4. 더 나은 추천을 위해 필요한 정보 1-2개를 자연스럽게 질문
5. 이모지를 적절히 사용하여 친근감 표현
6. 사용자가 부담스러워하지 않도록 선택형 질문 활용

위 내용을 바탕으로 도움이 되는 응답을 해주세요."""

        response = self.chat_model.invoke(question_prompt)
        return response.content.strip()
    
    def _generate_travel_recommendation_response(self, state: ChatbotState) -> str:
        """여행 추천을 위한 응답 생성"""
        
        # 검색 결과 컨텍스트
        search_context = ""
        if state.search_results:
            search_context = "검색된 제주도 정보:\n"
            for i, result in enumerate(state.search_results[:3], 1):
                name = result['metadata'].get('name', 'N/A')
                category = result['metadata'].get('category', '').upper()
                address = result['metadata'].get('address', 'N/A')
                phone = result['metadata'].get('phone', '')
                tags = result['metadata'].get('tags', '')
                
                search_context += f"{i}. [{category}] {name}\n"
                search_context += f"   📍 {address}\n"
                if phone:
                    search_context += f"   📞 {phone}\n"
                if tags:
                    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()][:3]
                    if tag_list:
                        search_context += f"   🏷️ {', '.join(tag_list)}\n"
                search_context += "\n"
        
        # 사용자 프로필 컨텍스트
        profile_context = f"""👥 동행자: {state.user_profile.travel_companions or '미정'}
📅 여행기간: {state.user_profile.duration or '미정'}
💰 예산: {state.user_profile.budget or '미정'}
❤️ 관심사: {', '.join(state.user_profile.interests) if state.user_profile.interests else '미정'}
🚗 교통: {state.user_profile.transportation or '미정'}
🏨 숙박선호: {state.user_profile.accommodation_type or '미정'}"""
        
        # 대화 히스토리 컨텍스트 추가
        conversation_context = ""
        if state.conversation_history and len(state.conversation_history) > 0:
            recent_messages = state.conversation_history[-5:]  # 최근 5개 메시지
            conversation_context = "\n이전 대화 맥락:\n"
            for msg in recent_messages:
                # interactive_demo의 대화 히스토리 구조에 맞게 수정
                if isinstance(msg, dict):
                    if "user_message" in msg and "bot_response" in msg:
                        conversation_context += f"- 사용자: {msg['user_message'][:100]}...\n"
                        conversation_context += f"- 상담사: {msg['bot_response'][:100]}...\n"
                    elif "role" in msg:
                        role = "사용자" if msg["role"] == "user" else "상담사"
                        conversation_context += f"- {role}: {msg['content'][:150]}...\n"
        
        # 의도 확인 - 일정 생성 요청인지 확인
        intent = state.analysis_result.get("intent", "general_info")
        
        if intent == "plan_trip":
            # 일정 생성 전용 오르미 프롬프트 사용
            recommendation_prompt = f"""System:
당신은 '오르미'라는 개인 맞춤형 여행 일정 추천 챗봇입니다. 사용자가 제주도 여행을 준비할 수 있도록 도와주세요. 다음 절차를 따르되, 말투는 친절하고 자연스럽게 대화체를 사용해야 하며, 불필요한 내부 추론(Cot reasoning)은 출력하지 마세요. 모든 장소 정보는 내부 DB에서만 조회해 정확성을 확인하세요.

1) 요청 요약  
   - 사용자의 여행 목적, 기간, 인원, 관심사 등을 한 문장으로 정리하되, 결과는 출력하지 않고 내부 참고용으로만 사용합니다.

2) 추가 정보 확인
   - 일정 추천에 꼭 필요한 정보(예: 예산, 교통 수단 선호, 숙소 종류)가 부족하면 최대 2가지 질문을 대화체로 자연스럽게 확인합니다.

3) CoT 기반 초안 생성  
   - 이 단계에서는 "단계 1: 주요 활동/장소 선정 → 단계 2: 이동 수단 및 시간 배분 → 단계 3: 예산·시간 검토"의 순서로 내부적으로 일정 초안을 구성하되, 이 내용은 사용자에게 출력하지 않습니다.

4) 세부 일정 완성
   - 일자별 아침·점심·저녁 활동과 장소, 예상 소요 시간, 교통수단을 표 형태로 제시합니다.
   - 친절하고 말로 설명한 뒤, 표로 정리해 보여주세요.

 | 일자 | 아침                      | 점심                 | 저녁                     |
   |-----|----------------------------|----------------------|---------------------------|

5) DB 검증  
   - 모든 장소 정보(운영 시간·주소·입장료 등)는 반드시 내부 DB에서 조회하고, "DB조회" 태그를 붙여 사용자에게 명확히 보여줍니다.

사용자 프로필:
{profile_context}

{conversation_context}

문서 내용:
{search_context}

질문:
{state.user_message}

답변:"""
        else:
            # 기존 일반 추천 프롬프트 사용
            recommendation_prompt = f"""제주도 여행 전문가로서 개인화된 추천을 제공해주세요.

사용자 질문: "{state.user_message}"

사용자 프로필:
{profile_context}

{conversation_context}

{search_context}

응답 가이드라인:
- 이전 대화의 맥락과 연결된 자연스러운 대화
- 사용자의 프로필과 관심사를 고려한 개인화된 추천
- 검색된 장소들의 특징과 매력을 구체적으로 설명
- 실용적인 정보(주소, 연락처, 이용팁) 포함
- 동행자와 기간을 고려한 일정 제안
- 자연스럽고 친근한 대화체
- 적절한 이모지 사용
- 사용자의 이전 질문이나 관심사에 대한 연속성 유지

위 내용을 바탕으로 도움이 되는 추천을 해주세요."""

        response = self.chat_model.invoke(recommendation_prompt)
        return response.content.strip()
    
    def chat_with_context(
        self, 
        user_message: str, 
        session_id: str = "interactive", 
        conversation_history: List[Dict] = None,
        existing_profile: Dict = None,
        profile_completion: int = 0
    ) -> Tuple[str, float]:
        """컨텍스트 기반 챗 메서드 (대화 히스토리와 프로필 정보 활용)"""
        print(f"\n💬 사용자: {user_message}")
        
        # 초기 상태 생성
        initial_state = ChatbotState(user_message=user_message)
        
        # 기존 프로필 정보 로드
        if existing_profile:
            try:
                # Dict를 UserProfile 객체로 변환
                initial_state.user_profile = UserProfile.from_dict(existing_profile)
                print(f"📋 기존 프로필 로드됨: 완성도 {profile_completion}%")
            except Exception as e:
                print(f"⚠️ 프로필 로드 실패: {e}, 새 프로필 생성")
                initial_state.user_profile = UserProfile()
        elif hasattr(self, 'session_profiles') and session_id in self.session_profiles:
            initial_state.user_profile = self.session_profiles[session_id]
        
        # 대화 히스토리 추가
        if conversation_history:
            # 외부에서 받은 대화 히스토리를 내부 형식으로 변환
            converted_history = []
            for msg in conversation_history:
                if isinstance(msg, dict) and "role" in msg:
                    # FastAPI 형식: {"role": "user/assistant", "content": "...", "timestamp": "..."}
                    converted_history.append({
                        "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                        "user_message": msg["content"] if msg["role"] == "user" else "",
                        "bot_response": msg["content"] if msg["role"] == "assistant" else "",
                        "profile_completion": 0.0,
                        "needs_more_info": False
                    })
            initial_state.conversation_history = converted_history
            print(f"💬 대화 히스토리 로드됨: {len(conversation_history)}개 메시지")
        
        # 멀티 에이전트 실행
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = self.graph.invoke(initial_state, config)
            
            # LangGraph 반환 타입 처리 (dict 또는 객체)
            if isinstance(final_state, dict):
                user_profile = final_state.get('user_profile', UserProfile())
                response = final_state.get('response', "응답을 생성하지 못했습니다.")
            else:
                user_profile = final_state.user_profile if hasattr(final_state, 'user_profile') else UserProfile()
                response = final_state.response if hasattr(final_state, 'response') else "응답을 생성하지 못했습니다."
            
            # 프로필 저장 (세션별)
            if not hasattr(self, 'session_profiles'):
                self.session_profiles = {}
            self.session_profiles[session_id] = user_profile
            
            completion_rate = user_profile.completion_rate()
            
            print(f"✅ 응답 생성 완료 (완성도: {completion_rate:.1%})")
            return response, completion_rate
            
        except Exception as e:
            error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)} 🙏"
            print(f"❌ 오류: {error_msg}")
            return error_msg, 0.0

    def chat(self, user_message: str, session_id: str = "interactive") -> Tuple[str, float]:
        """기존 호환성을 위한 메인 챗 메서드"""
        return self.chat_with_context(user_message, session_id)

def interactive_demo():
    """인터랙티브 데모 실행"""
    print("🌟 제주도 여행 챗봇 인터랙티브 데모")
    print("=" * 60)
    print("💡 팁: 'quit' 또는 'exit' 입력시 종료됩니다.")
    print("💡 팁: 동행자, 기간, 예산, 관심사를 알려주시면 더 정확한 추천을 받을 수 있어요!")
    print("=" * 60)
    
    try:
        # 챗봇 초기화
        chatbot = InteractiveJejuChatbot()
        session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n✅ 챗봇 준비 완료! (세션 ID: {session_id})")
        print("🚀 제주도 여행에 대해 무엇이든 물어보세요!\n")
        
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 You: ").strip()
                
                # 종료 명령어 확인
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("\n👋 제주도 여행 챗봇을 이용해주셔서 감사합니다!")
                    print("🌴 멋진 제주 여행 되세요! ✨")
                    break
                
                if not user_input:
                    print("💭 메시지를 입력해주세요!")
                    continue
                
                # 챗봇 응답 (대화 히스토리 활용)
                response, completion = chatbot.chat_with_context(user_input, session_id)
                
                # 응답 출력
                print(f"\n🤖 제주도 챗봇: {response}")
                
                # 프로필 완성도 표시
                if completion > 0:
                    print(f"\n📊 프로필 완성도: {completion:.1%} {'🌟' * int(completion * 5)}")
                    
                print("\n" + "─" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Ctrl+C로 종료합니다. 좋은 여행 되세요!")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("다시 시도해주세요.")
    
    except Exception as e:
        print(f"❌ 챗봇 초기화 실패: {e}")
        print("💡 Upstage API 키나 ChromaDB 상태를 확인해주세요.")

def test_context_conversation():
    """컨텍스트 기반 대화 테스트"""
    print("🧪 컨텍스트 기반 대화 테스트 시작")
    print("=" * 60)
    
    try:
        chatbot = InteractiveJejuChatbot()
        session_id = "context_test"
        
        # 테스트 시나리오: 3박4일 여자친구와 여행 → 음식 제약사항 추가 → 수정 요청
        test_conversations = [
            "여자친구랑 3박4일로 제주도 여행 가려고 해",
            "우유랑 치즈가 들어간 음식은 못 먹어",
            "아까 3박4일이라고 했는데 왜 2박3일로 바뀌었어?"
        ]
        
        conversation_history = []
        user_profile = {}
        completion = 0.0
        
        for i, message in enumerate(test_conversations, 1):
            print(f"\n🔸 테스트 {i}")
            print(f"👤 사용자: {message}")
            
            # 컨텍스트 기반 응답
            response, completion = chatbot.chat_with_context(
                user_message=message,
                session_id=session_id,
                conversation_history=conversation_history,
                existing_profile=user_profile,
                profile_completion=int(completion * 100)
            )
            
            print(f"🤖 챗봇: {response[:200]}...")
            print(f"📊 완성도: {completion:.1%}")
            
            # 대화 히스토리 업데이트 (FastAPI 형식으로)
            conversation_history.extend([
                {"role": "user", "content": message, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            ])
            
            # 프로필 업데이트
            if hasattr(chatbot, 'session_profiles') and session_id in chatbot.session_profiles:
                user_profile = chatbot.session_profiles[session_id].to_dict()
            
            print("-" * 40)
        
        print("\n✅ 컨텍스트 테스트 완료!")
        print("💡 대화 맥락이 올바르게 유지되는지 확인해보세요.")
        
    except Exception as e:
        print(f"❌ 컨텍스트 테스트 실패: {e}")

def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-context":
        test_context_conversation()
    else:
        interactive_demo()

if __name__ == "__main__":
    main() 