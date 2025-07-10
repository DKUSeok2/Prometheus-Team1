"""
쿼리 분석 에이전트
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent

class QueryAnalyzer(BaseAgent):
    """사용자 쿼리를 분석하여 의도와 카테고리를 파악하는 에이전트"""
    
    def __init__(self):
        super().__init__("QueryAnalyzer")
        self.categories = ["음식", "숙소", "관광지", "행사"]
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """쿼리 분석 처리"""
        
        # 시스템 프롬프트 정의
        system_prompt = """
        당신은 제주도 여행 관련 쿼리를 분석하는 전문가입니다.
        사용자의 질문을 분석하여 다음 정보를 파악해주세요:
        
        1. 의도 (intent): 일정추천, 정보검색, 예약도움, 항공권검색, 일반대화 중 하나
        2. 카테고리 (categories): 음식, 숙소, 관광지, 행사, 항공 중 관련된 것들 (복수 가능)
        3. 키워드 (keywords): 검색에 중요한 핵심 키워드들
        4. 위치 (location): 제주도 내 특정 지역이 언급된 경우
        5. 여행 스타일 (travel_style): 가족여행, 커플여행, 혼자여행, 친구여행 등
        
        **항공권 검색 기준:**
        - 항공편, 비행기, 항공권, 티켓, 가는법, 교통 등의 키워드가 포함된 경우
        - 출발지와 도착지가 언급된 경우 (예: 서울에서 제주도)
        - 날짜와 함께 여행 계획을 물어보는 경우
        
        응답은 반드시 다음 형식으로 해주세요:
        의도: [의도]
        카테고리: [카테고리1, 카테고리2, ...]
        키워드: [키워드1, 키워드2, ...]
        위치: [위치정보 또는 없음]
        여행스타일: [여행스타일 또는 없음]
        """
        
        user_message = f"사용자 질문: {query}"
        
        try:
            # LLM으로 쿼리 분석
            response = self.llm.invoke(f"{system_prompt}\n\n{user_message}")
            
            # 응답 파싱
            analysis = self._parse_analysis_response(response)
            
            return self.format_response(
                content="쿼리 분석 완료",
                metadata={
                    "query": query,
                    "analysis": analysis,
                    "raw_response": response
                }
            )
        
        except Exception as e:
            return self.format_response(
                content=f"쿼리 분석 중 오류 발생: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 구조화된 정보로 변환"""
        analysis = {
            "intent": "일반대화",
            "categories": [],
            "keywords": [],
            "location": None,
            "travel_style": None
        }
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('의도:'):
                analysis["intent"] = line.replace('의도:', '').strip()
            elif line.startswith('카테고리:'):
                categories_str = line.replace('카테고리:', '').strip()
                if categories_str and categories_str != "없음":
                    # 괄호와 쉼표로 분리
                    categories_str = categories_str.replace('[', '').replace(']', '')
                    analysis["categories"] = [cat.strip() for cat in categories_str.split(',') if cat.strip()]
            elif line.startswith('키워드:'):
                keywords_str = line.replace('키워드:', '').strip()
                if keywords_str and keywords_str != "없음":
                    keywords_str = keywords_str.replace('[', '').replace(']', '')
                    analysis["keywords"] = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            elif line.startswith('위치:'):
                location = line.replace('위치:', '').strip()
                if location and location != "없음":
                    analysis["location"] = location
            elif line.startswith('여행스타일:'):
                travel_style = line.replace('여행스타일:', '').strip()
                if travel_style and travel_style != "없음":
                    analysis["travel_style"] = travel_style
        
        return analysis
    
    def get_search_categories(self, analysis: Dict[str, Any]) -> List[str]:
        """분석 결과에서 검색할 카테고리 목록 반환"""
        categories = analysis.get("categories", [])
        
        # 카테고리가 명시되지 않은 경우 키워드로 추론
        if not categories:
            keywords = analysis.get("keywords", [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if any(food_word in keyword_lower for food_word in ["음식", "맛집", "카페", "식당", "레스토랑"]):
                    categories.append("음식")
                elif any(hotel_word in keyword_lower for hotel_word in ["숙소", "호텔", "펜션", "리조트", "게스트하우스"]):
                    categories.append("숙소")
                elif any(tour_word in keyword_lower for tour_word in ["관광", "여행", "명소", "체험", "관광지"]):
                    categories.append("관광지")
                elif any(event_word in keyword_lower for event_word in ["행사", "축제", "이벤트", "공연"]):
                    categories.append("행사")
        
        # 여전히 카테고리가 없으면 모든 카테고리 검색
        if not categories:
            categories = self.categories
        
        return list(set(categories))  # 중복 제거
