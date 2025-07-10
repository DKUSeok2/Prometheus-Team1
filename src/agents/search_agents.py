"""
카테고리별 검색 에이전트들
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..database import chromadb_manager

class SearchAgent(BaseAgent):
    """벡터 검색을 수행하는 기본 검색 에이전트"""
    
    def __init__(self, category: str):
        super().__init__(f"{category}SearchAgent")
        self.category = category
        
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """검색 처리"""
        try:
            # ChromaDB에서 카테고리별 검색
            results = chromadb_manager.search(
                query=query,
                category_filter=self.category,
                n_results=context.get("max_results", 5) if context else 5
            )
            
            # 검색 결과 포맷팅
            formatted_results = chromadb_manager._format_search_results(results)
            
            return self.format_response(
                content=f"{self.category} 검색 완료 - {len(formatted_results)}개 결과",
                metadata={
                    "category": self.category,
                    "query": query,
                    "results": formatted_results,
                    "total_results": len(formatted_results)
                }
            )
            
        except Exception as e:
            return self.format_response(
                content=f"{self.category} 검색 중 오류 발생: {str(e)}",
                metadata={"error": str(e), "category": self.category}
            )

class FoodSearchAgent(SearchAgent):
    """음식 검색 에이전트"""
    
    def __init__(self):
        super().__init__("음식")
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """음식 검색 특화 처리"""
        # 음식 관련 키워드 보강
        enhanced_query = self._enhance_food_query(query)
        
        # 기본 검색 수행
        result = super().process(enhanced_query, context)
        
        # 음식 특화 정보 추가
        if result["metadata"].get("results"):
            result["metadata"]["specialized_info"] = "음식 맛집 정보"
            
        return result
    
    def _enhance_food_query(self, query: str) -> str:
        """음식 검색을 위한 쿼리 보강"""
        food_keywords = ["맛집", "음식", "식당", "카페", "레스토랑"]
        
        if not any(keyword in query for keyword in food_keywords):
            return f"{query} 맛집"
        return query

class AccommodationSearchAgent(SearchAgent):
    """숙소 검색 에이전트"""
    
    def __init__(self):
        super().__init__("숙소")
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """숙소 검색 특화 처리"""
        enhanced_query = self._enhance_accommodation_query(query)
        result = super().process(enhanced_query, context)
        
        if result["metadata"].get("results"):
            result["metadata"]["specialized_info"] = "숙박 시설 정보"
            
        return result
    
    def _enhance_accommodation_query(self, query: str) -> str:
        """숙소 검색을 위한 쿼리 보강"""
        accommodation_keywords = ["숙소", "호텔", "펜션", "리조트", "게스트하우스"]
        
        if not any(keyword in query for keyword in accommodation_keywords):
            return f"{query} 숙소"
        return query

class TourismSearchAgent(SearchAgent):
    """관광지 검색 에이전트"""
    
    def __init__(self):
        super().__init__("관광지")
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """관광지 검색 특화 처리"""
        enhanced_query = self._enhance_tourism_query(query)
        result = super().process(enhanced_query, context)
        
        if result["metadata"].get("results"):
            result["metadata"]["specialized_info"] = "관광 명소 정보"
            
        return result
    
    def _enhance_tourism_query(self, query: str) -> str:
        """관광지 검색을 위한 쿼리 보강"""
        tourism_keywords = ["관광지", "명소", "여행", "체험", "볼거리"]
        
        if not any(keyword in query for keyword in tourism_keywords):
            return f"{query} 관광지"
        return query

class EventSearchAgent(SearchAgent):
    """행사 검색 에이전트"""
    
    def __init__(self):
        super().__init__("행사")
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """행사 검색 특화 처리"""
        enhanced_query = self._enhance_event_query(query)
        result = super().process(enhanced_query, context)
        
        if result["metadata"].get("results"):
            result["metadata"]["specialized_info"] = "행사 및 축제 정보"
            
        return result
    
    def _enhance_event_query(self, query: str) -> str:
        """행사 검색을 위한 쿼리 보강"""
        event_keywords = ["행사", "축제", "이벤트", "공연", "문화"]
        
        if not any(keyword in query for keyword in event_keywords):
            return f"{query} 행사"
        return query

# 에이전트 인스턴스들
food_agent = FoodSearchAgent()
accommodation_agent = AccommodationSearchAgent()
tourism_agent = TourismSearchAgent()
event_agent = EventSearchAgent()

# 카테고리별 에이전트 매핑
CATEGORY_AGENTS = {
    "음식": food_agent,
    "숙소": accommodation_agent,
    "관광지": tourism_agent,
    "행사": event_agent
}
