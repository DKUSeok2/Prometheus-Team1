"""
제주도 여행 챗봇 최종 버전 (FastAPI 연결 준비)
- Solar Pro API 기반 고품질 대화
- 사용자 프로필 점진적 구축
- 멀티 에이전트 LangGraph 시스템
- 조건부 정보 수집 및 추천
"""

import chromadb
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        total_fields = 6
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

@dataclass 
class ChatResponse:
    """챗봇 응답 데이터 클래스"""
    response: str
    needs_more_info: bool
    profile_completion: float
    follow_up_questions: List[str]
    user_profile: Dict
    analysis_confidence: float
    timestamp: str

class JejuTravelChatbot:
    """제주도 여행 챗봇 최종 버전"""
    
    def __init__(self):
        """챗봇 초기화"""
        logger.info("🌟 제주도 여행 챗봇 초기화 중...")
        
        try:
            # Upstage 임베딩 모델
            self.embeddings = UpstageEmbeddings(
                api_key=UPSTAGE_API_KEY,
                model=EMBEDDING_MODEL
            )
            logger.info(f"✅ Upstage 임베딩 모델 로드: {EMBEDDING_MODEL}")
            
            # Upstage Solar Pro API
            self.chat_model = ChatUpstage(
                api_key=UPSTAGE_API_KEY,
                model="solar-pro",
                temperature=0.7
            )
            logger.info("✅ Upstage Solar Pro API 연결 성공")
            
            # ChromaDB 클라이언트
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            logger.info(f"✅ ChromaDB 컬렉션 연결: {COLLECTION_NAME}")
            
            # 메모리 체크포인트
            self.memory = MemorySaver()
            
            # 멀티 에이전트 그래프 구성
            self.graph = self._build_agent_graph()
            logger.info("✅ 멀티 에이전트 그래프 구성 완료")
            
            # 세션별 프로필 저장소
            self.session_profiles = {}
            
        except Exception as e:
            logger.error(f"❌ 챗봇 초기화 실패: {e}")
            raise
    
    def _build_agent_graph(self) -> StateGraph:
        """멀티 에이전트 그래프 구성"""
        graph = StateGraph(ChatbotState)
        
        # 에이전트 노드 추가
        graph.add_node("query_analyzer", self.analyze_query_with_solar_pro)
        graph.add_node("profile_updater", self.update_user_profile)  
        graph.add_node("info_checker", self.check_missing_info)
        graph.add_node("search_agent", self.search_travel_data)
        graph.add_node("response_generator", self.generate_response_with_solar_pro)
        
        # 조건부 엣지 설정
        graph.add_edge("query_analyzer", "profile_updater")
        graph.add_edge("profile_updater", "info_checker")
        
        def should_ask_more_info(state: ChatbotState) -> str:
            """정보 부족 여부에 따른 분기"""
            return "response_generator" if state.needs_more_info else "search_agent"
        
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
        logger.info("🔍 Solar Pro 쿼리 분석 중...")
        
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

            response = self.chat_model.invoke(analysis_prompt)
            analysis_response = response.content.strip()
            
            # JSON 파싱 (내부 처리)
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
                logger.warning("JSON 파싱 실패, 키워드 기반 분석으로 전환")  # 구체적 에러 내용 숨김
                
                # 백업 키워드 분석
                intent = self._classify_intent_by_keywords(state.user_message, analysis_response)
                travel_info = self._extract_info_by_keywords(state.user_message, analysis_response)
                confidence = 0.7
                user_providing_info = any(word in state.user_message.lower() 
                                        for word in ["명", "박", "일", "가족", "친구", "혼자", "만원", "원"])
                reasoning = "키워드 기반 분석"
            
            state.analysis_result = {
                "intent": intent,
                "travel_info": travel_info,
                "confidence": confidence,
                "user_providing_info": user_providing_info,
                "reasoning": reasoning
                # "solar_analysis": analysis_response  # 사용자 노출 방지를 위해 제거
            }
            
            logger.info(f"✅ 분석 결과: {intent} (확신도: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"쿼리 분석 실패")  # 구체적 에러 내용 숨김
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
        interest_mapping = {
            "음식": ["음식", "맛집", "먹거리"],
            "바다": ["바다", "해변", "해수욕장"],
            "산": ["산", "한라산", "등산", "오름"],
            "문화": ["문화", "역사", "박물관"],
            "액티비티": ["액티비티", "체험", "스포츠"]
        }
        
        for interest, keywords in interest_mapping.items():
            if any(keyword in text for keyword in keywords):
                interests.append(interest)
        
        info["interests"] = interests
        return info
    
    def update_user_profile(self, state: ChatbotState) -> ChatbotState:
        """2단계: 사용자 프로필 업데이트"""
        logger.info("👤 사용자 프로필 업데이트 중...")
        
        if state.analysis_result:
            travel_info = state.analysis_result.get("travel_info", {})
            
            # 프로필 정보 업데이트 (기존 정보 유지)
            for field in ["companions", "duration", "budget", "transportation"]:
                value = travel_info.get(field)
                if value and value != "불명":
                    setattr(state.user_profile, 
                           f"travel_{field}" if field == "companions" else field, 
                           value)
            
            # 숙박 선호도
            if travel_info.get("accommodation_preference") and travel_info["accommodation_preference"] != "불명":
                state.user_profile.accommodation_type = travel_info["accommodation_preference"]
            
            # 관심사 추가 (중복 제거)
            if travel_info.get("interests"):
                for interest in travel_info["interests"]:
                    if interest not in state.user_profile.interests:
                        state.user_profile.interests.append(interest)
        
        completion = state.user_profile.completion_rate()
        logger.info(f"프로필 완성도: {completion:.1%}")
        
        return state
    
    def check_missing_info(self, state: ChatbotState) -> ChatbotState:
        """3단계: 필요 정보 부족 여부 확인"""
        logger.info("🔍 필요 정보 확인 중...")
        
        missing_info = state.user_profile.get_missing_info()
        intent = state.analysis_result.get("intent", "general_info")
        
        # 구체적인 요청에 대해 정보 수집 우선
        needs_specific_info = intent in ["plan_trip", "find_food", "find_hotel", "find_attraction"]
        completion_rate = state.user_profile.completion_rate()
        
        # 정보 수집 필요 조건
        state.needs_more_info = (
            needs_specific_info and 
            completion_rate < 0.7 and  # 70% 미만일 때 추가 정보 수집
            len(missing_info) > 0 and
            intent != "greeting"
        )
        
        if state.needs_more_info:
            state.follow_up_questions = self._generate_priority_questions(missing_info, intent)
            logger.info(f"추가 정보 필요: {missing_info}")
        else:
            logger.info(f"정보 충분 (완성도: {completion_rate:.1%})")
        
        return state
    
    def _generate_priority_questions(self, missing_info: List[str], intent: str) -> List[str]:
        """우선순위 기반 질문 생성"""
        questions = []
        
        # 의도별 우선순위 맵핑
        priority_map = {
            "plan_trip": ["동행자", "여행기간", "예산", "관심사"],
            "find_food": ["관심사", "동행자", "예산"],
            "find_hotel": ["동행자", "여행기간", "예산"],
            "find_attraction": ["관심사", "동행자", "여행기간"]
        }
        
        priority_info = priority_map.get(intent, ["동행자", "여행기간"])
        
        question_templates = {
            "동행자": "누구와 함께 여행을 가시나요? (가족/친구/연인/혼자)",
            "여행기간": "몇 박 며칠 일정으로 계획하고 계시나요?",
            "예산": "여행 예산은 어느 정도로 생각하고 계시나요?",
            "관심사": "어떤 것에 가장 관심이 있으신가요? (음식/바다/산/문화/액티비티 등)"
        }
        
        for info_type in priority_info:
            if info_type in missing_info:
                questions.append(question_templates[info_type])
        
        return questions[:2]  # 최대 2개 질문
    
    def search_travel_data(self, state: ChatbotState) -> ChatbotState:
        """4단계: 여행 데이터 검색"""
        logger.info("🔎 여행 데이터 검색 중...")
        
        try:
            # 검색 쿼리 생성
            search_query = state.user_message
            
            # 프로필 기반 쿼리 강화
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
            logger.info(f"검색 결과: {len(search_results)}개")
            
        except Exception as e:
            logger.error("검색 실패")  # 구체적 에러 내용 숨김
            state.search_results = []
        
        return state
    
    def generate_response_with_solar_pro(self, state: ChatbotState) -> ChatbotState:
        """5단계: Solar Pro 응답 생성"""
        logger.info("💬 Solar Pro 응답 생성 중...")
        
        try:
            if state.needs_more_info:
                response = self._generate_info_collection_response(state)
            else:
                response = self._generate_travel_recommendation_response(state)
            
            state.response = response
            logger.info("Solar Pro 응답 생성 완료")
            
        except Exception as e:
            logger.error("응답 생성 실패")  # 구체적 에러 내용 숨김
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
        profile_summary = []
        if state.user_profile.travel_companions:
            profile_summary.append(f"👥 {state.user_profile.travel_companions}과 함께")
        if state.user_profile.duration:
            profile_summary.append(f"📅 {state.user_profile.duration} 일정")
        if state.user_profile.interests:
            profile_summary.append(f"❤️ {', '.join(state.user_profile.interests)} 관심")
        
        profile_text = ", ".join(profile_summary) if profile_summary else "처음 상담"
        
        # 대화 히스토리 컨텍스트 추가
        conversation_context = ""
        if state.conversation_history and len(state.conversation_history) > 0:
            recent_messages = state.conversation_history[-3:]  # 최근 3개 메시지만
            conversation_context = "\n이전 대화:\n"
            for msg in recent_messages:
                role = "사용자" if msg["role"] == "user" else "상담사"
                conversation_context += f"- {role}: {msg['content'][:100]}...\n"
        
        question_prompt = f"""제주도 여행 전문 상담사로서 사용자의 여행 계획을 도와주세요.

현재까지 파악된 정보: {profile_text}

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
        session_id: str = "default", 
        conversation_history: List[Dict] = None,
        existing_profile: Dict = None,
        profile_completion: int = 0
    ) -> ChatResponse:
        """컨텍스트 기반 챗 메서드 (대화 히스토리와 프로필 정보 활용)"""
        logger.info(f"사용자 메시지 수신: {user_message}")
        
        # 초기 상태 생성
        initial_state = ChatbotState(user_message=user_message)
        
        # 기존 프로필 정보 로드
        if existing_profile:
            try:
                # Dict를 UserProfile 객체로 변환
                initial_state.user_profile = UserProfile.from_dict(existing_profile)
                logger.info(f"📋 기존 프로필 로드됨: 완성도 {profile_completion}%")
            except Exception as e:
                logger.warning(f"프로필 로드 실패: {e}, 새 프로필 생성")
                initial_state.user_profile = UserProfile()
        
        # 대화 히스토리 추가
        if conversation_history:
            initial_state.conversation_history = conversation_history
            logger.info(f"💬 대화 히스토리 로드됨: {len(conversation_history)}개 메시지")
        
        # 멀티 에이전트 실행
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = self.graph.invoke(initial_state, config)
            
            # LangGraph 반환 타입 처리
            if isinstance(final_state, dict):
                user_profile = final_state.get('user_profile', UserProfile())
                response = final_state.get('response', "응답을 생성하지 못했습니다.")
                needs_more_info = final_state.get('needs_more_info', False)
                follow_up_questions = final_state.get('follow_up_questions', [])
                analysis_result = final_state.get('analysis_result', {})
            else:
                user_profile = final_state.user_profile
                response = final_state.response
                needs_more_info = final_state.needs_more_info
                follow_up_questions = final_state.follow_up_questions
                analysis_result = final_state.analysis_result
            
            # 세션 프로필 저장
            self.session_profiles[session_id] = user_profile
            
            # 응답 객체 생성
            chat_response = ChatResponse(
                response=response,
                needs_more_info=needs_more_info,
                profile_completion=user_profile.completion_rate(),
                follow_up_questions=follow_up_questions,
                user_profile=user_profile.to_dict(),
                analysis_confidence=analysis_result.get('confidence', 0.0),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"응답 생성 완료 (완성도: {chat_response.profile_completion:.1%})")
            return chat_response
            
        except Exception as e:
            logger.error("챗봇 처리 오류")
            return ChatResponse(
                response="죄송합니다. 오류가 발생했습니다. 다시 시도해주세요. 🙏",
                needs_more_info=False,
                profile_completion=0.0,
                follow_up_questions=[],
                user_profile={},
                analysis_confidence=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def chat(self, user_message: str, session_id: str = "default") -> ChatResponse:
        """기존 호환성을 위한 메인 챗 메서드"""
        return self.chat_with_context(user_message, session_id)
    
    def get_user_profile(self, session_id: str) -> Dict:
        """세션별 사용자 프로필 조회"""
        if session_id in self.session_profiles:
            return self.session_profiles[session_id].to_dict()
        return UserProfile().to_dict()
    
    def update_user_profile_manual(self, session_id: str, profile_data: Dict) -> bool:
        """사용자 프로필 수동 업데이트"""
        try:
            if session_id not in self.session_profiles:
                self.session_profiles[session_id] = UserProfile()
            
            profile = self.session_profiles[session_id]
            for key, value in profile_data.items():
                if hasattr(profile, key) and value:
                    setattr(profile, key, value)
            
            return True
        except Exception as e:
            logger.error("프로필 업데이트 실패")  # 구체적 에러 내용 숨김
            return False
    
    def reset_session(self, session_id: str) -> bool:
        """세션 초기화"""
        try:
            if session_id in self.session_profiles:
                del self.session_profiles[session_id]
            return True
        except Exception as e:
            logger.error("세션 초기화 실패")  # 구체적 에러 내용 숨김
            return False

# 테스트용 메인 함수
def main():
    """테스트 실행"""
    try:
        chatbot = JejuTravelChatbot()
        
        # 테스트 대화
        test_messages = [
            "안녕하세요! 제주도 여행을 계획하고 있어요.",
            "친구와 2박3일로 가려고 해요. 바다를 좋아해요!",
            "예산은 보통 정도로 생각하고 있어요.",
            "성산일출봉 근처 맛집 추천해주세요."
        ]
        
        session_id = "test_session"
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n[테스트 {i}]")
            print("=" * 50)
            
            response = chatbot.chat(message, session_id)
            
            print(f"👤 사용자: {message}")
            print(f"🤖 챗봇: {response.response}")
            print(f"📊 완성도: {response.profile_completion:.1%}")
            print(f"🔍 추가정보필요: {response.needs_more_info}")
            
            if response.follow_up_questions:
                print(f"❓ 후속질문: {response.follow_up_questions}")
        
        print("\n✅ 최종 챗봇 테스트 완료!")
        
    except Exception as e:
        logger.error("테스트 실패")  # 구체적 에러 내용 숨김

if __name__ == "__main__":
    main() 