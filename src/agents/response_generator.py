"""
응답 생성 에이전트
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent

class ResponseGenerator(BaseAgent):
    """검색 결과를 바탕으로 사용자 친화적인 응답을 생성하는 에이전트"""
    
    def __init__(self):
        super().__init__("ResponseGenerator")
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """응답 생성 처리"""
        try:
            search_results = context.get("search_results", {}) if context else {}
            analysis = context.get("analysis", {}) if context else {}
            chat_history = context.get("chat_history", []) if context else []
            
            # 응답 생성
            response = self._generate_response(query, search_results, analysis, chat_history)
            
            return self.format_response(
                content=response,
                metadata={
                    "query": query,
                    "search_results_count": sum(len(results) for results in search_results.values()),
                    "categories_searched": list(search_results.keys()),
                    "recommendations": self._extract_recommendations(search_results)
                }
            )
            
        except Exception as e:
            return self.format_response(
                content=f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _generate_response(self, query: str, search_results: Dict[str, List], analysis: Dict[str, Any], chat_history: List[Dict]) -> str:
        """실제 응답 생성"""
        
        # 이전 대화 맥락 구성
        history_context = self._build_history_context(chat_history)
        
        # 검색 결과 요약
        search_summary = self._summarize_search_results(search_results)
        
        # 시스템 프롬프트
        system_prompt = f"""
        당신은 제주도 여행 전문 가이드입니다. 사용자의 질문에 대해 친근하고 도움이 되는 답변을 제공해주세요.
        
        사용자 질문: {query}
        
        분석된 의도: {analysis.get('intent', '정보검색')}
        분석된 키워드: {', '.join(analysis.get('keywords', []))}
        위치 정보: {analysis.get('location', '특정 위치 없음')}
        여행 스타일: {analysis.get('travel_style', '명시되지 않음')}
        
        이전 대화 맥락:
        {history_context}
        
        검색 결과:
        {search_summary}
        
        다음 지침을 따라 답변해주세요:
        1. 친근하고 따뜻한 톤으로 답변
        2. 검색 결과를 바탕으로 구체적인 추천 제공
        3. 각 추천에 대해 간단한 설명과 특징 포함
        4. 이전 대화 내용을 고려하여 일관된 답변
        5. 질문이 명확하지 않으면 추가 질문으로 명확화 유도
        6. 제주도 여행 팁이나 주의사항도 함께 제공
        
        응답은 자연스럽고 도움이 되는 형태로 작성해주세요.
        """
        
        try:
            response = self.llm.invoke(system_prompt)
            return response
        except Exception as e:
            return f"죄송합니다. 응답 생성 중 문제가 발생했습니다. 다시 질문해주시면 도와드리겠습니다."
    
    def _build_history_context(self, chat_history: List[Dict]) -> str:
        """대화 히스토리 맥락 구성"""
        if not chat_history:
            return "이번이 첫 대화입니다."
        
        context_parts = []
        for item in chat_history[-3:]:  # 최근 3개 대화만 사용
            if item.get("role") == "user":
                context_parts.append(f"사용자: {item.get('content', '')}")
            elif item.get("role") == "assistant":
                context_parts.append(f"어시스턴트: {item.get('content', '')[:100]}...")  # 너무 길면 요약
        
        return "\n".join(context_parts)
    
    def _summarize_search_results(self, search_results: Dict[str, List]) -> str:
        """검색 결과 요약"""
        if not search_results:
            return "검색 결과가 없습니다."
        
        summary_parts = []
        
        for category, results in search_results.items():
            if results:
                summary_parts.append(f"\n{category} 관련 정보:")
                for i, result in enumerate(results[:3], 1):  # 카테고리당 최대 3개
                    metadata = result.get("metadata", {})
                    name = metadata.get("이름") or metadata.get("title", "이름 없음")
                    address = metadata.get("주소") or metadata.get("roadaddress", "주소 없음")
                    intro = metadata.get("소개") or metadata.get("introduction", "")
                    
                    summary_parts.append(f"{i}. {name}")
                    summary_parts.append(f"   위치: {address}")
                    if intro:
                        summary_parts.append(f"   설명: {intro[:100]}...")
        
        return "\n".join(summary_parts) if summary_parts else "관련 정보를 찾을 수 없습니다."
    
    def _extract_recommendations(self, search_results: Dict[str, List]) -> List[Dict]:
        """검색 결과에서 추천 목록 추출"""
        recommendations = []
        
        for category, results in search_results.items():
            for result in results:
                metadata = result.get("metadata", {})
                recommendations.append({
                    "category": category,
                    "name": metadata.get("이름") or metadata.get("title", ""),
                    "address": metadata.get("주소") or metadata.get("roadaddress", ""),
                    "phone": metadata.get("전화번호", ""),
                    "description": metadata.get("소개") or metadata.get("introduction", ""),
                    "tags": metadata.get("태그") or metadata.get("alltag", ""),
                    "distance": result.get("distance", 0)
                })
        
        # 유사도 점수로 정렬
        recommendations.sort(key=lambda x: x["distance"])
        
        return recommendations
