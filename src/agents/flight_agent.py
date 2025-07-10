"""
항공권 검색 에이전트
"""
import re
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from ..flight.amadeus_client import amadeus_client
from ..flight.flight_types import FlightSearchResponse, FlightSearchError

class FlightSearchAgent:
    """항공권 검색 에이전트"""
    
    def __init__(self):
        self.client = amadeus_client
        
        # 항공권 관련 키워드
        self.flight_keywords = [
            "항공", "비행기", "항공편", "항공권", "티켓", "티케팅", 
            "출발", "도착", "여행", "가는법", "교통", "항공료", "운임"
        ]
        
        # 날짜 패턴
        self.date_patterns = [
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',  # 2024-01-01, 2024/01/01
            r'(\d{1,2})월\s*(\d{1,2})일',  # 1월 1일
            r'(\d{1,2})/(\d{1,2})',  # 1/1
            r'오늘|내일|모레|다음주|다음달',  # 상대적 날짜
        ]
    
    def is_flight_related_query(self, query: str) -> bool:
        """항공권 관련 쿼리인지 판단"""
        query_lower = query.lower()
        
        # 항공 키워드 확인
        has_flight_keyword = any(keyword in query_lower for keyword in self.flight_keywords)
        
        # 목적지가 제주도가 아닌 경우 (제주도 여행 챗봇이므로)
        has_other_destination = any(dest in query_lower for dest in [
            "서울", "부산", "인천", "대구", "광주", "도쿄", "오사카", "상하이", "베이징", "홍콩"
        ])
        
        # 날짜 패턴 확인
        has_date_pattern = any(re.search(pattern, query) for pattern in self.date_patterns)
        
        return has_flight_keyword or (has_other_destination and has_date_pattern)
    
    def extract_flight_info(self, query: str) -> Dict[str, Any]:
        """쿼리에서 항공편 정보 추출"""
        result = {
            "origin": None,
            "destination": None, 
            "departure_date": None,
            "return_date": None,
            "passengers": 1
        }
        
        # 출발지/도착지 추출
        locations = self._extract_locations(query)
        if locations:
            if len(locations) >= 2:
                result["origin"] = locations[0]
                result["destination"] = locations[1]
            elif len(locations) == 1:
                # 제주도 여행 챗봇이므로 기본적으로 제주도가 목적지
                if "제주" not in locations[0]:
                    result["origin"] = locations[0]
                    result["destination"] = "제주"
                else:
                    result["destination"] = locations[0]
        
        # 날짜 추출
        dates = self._extract_dates(query)
        if dates:
            result["departure_date"] = dates[0]
            if len(dates) > 1:
                result["return_date"] = dates[1]
        
        # 승객 수 추출
        passengers = self._extract_passenger_count(query)
        if passengers:
            result["passengers"] = passengers
        
        return result
    
    def _extract_locations(self, query: str) -> List[str]:
        """지명 추출"""
        locations = []
        
        # 지명 패턴
        location_patterns = [
            r'(서울|인천|김포|부산|김해|대구|광주|제주도?|제주)',
            r'(도쿄|나리타|하네다|오사카|간사이|후쿠오카|상하이|푸동|베이징|홍콩|타이베이|방콕|싱가포르)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query)
            locations.extend(matches)
        
        # 중복 제거 및 순서 유지
        seen = set()
        unique_locations = []
        for loc in locations:
            if loc not in seen:
                seen.add(loc)
                unique_locations.append(loc)
        
        return unique_locations
    
    def _extract_dates(self, query: str) -> List[str]:
        """날짜 추출"""
        dates = []
        
        # YYYY-MM-DD 형식
        date_matches = re.findall(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', query)
        for year, month, day in date_matches:
            dates.append(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
        
        # 상대적 날짜 처리
        today = date.today()
        if "오늘" in query:
            dates.append(today.isoformat())
        elif "내일" in query:
            dates.append((today + timedelta(days=1)).isoformat())
        elif "모레" in query:
            dates.append((today + timedelta(days=2)).isoformat())
        elif "다음주" in query:
            dates.append((today + timedelta(days=7)).isoformat())
        
        return dates
    
    def _extract_passenger_count(self, query: str) -> Optional[int]:
        """승객 수 추출"""
        # 숫자 + 명/인/사람 패턴
        passenger_patterns = [
            r'(\d+)\s*명',
            r'(\d+)\s*인',
            r'(\d+)\s*사람'
        ]
        
        for pattern in passenger_patterns:
            match = re.search(pattern, query)
            if match:
                return int(match.group(1))
        
        return None
    
    def search_flights(self, flight_info: Dict[str, Any]) -> Dict[str, Any]:
        """항공편 검색 수행"""
        try:
            origin = flight_info.get("origin")
            destination = flight_info.get("destination")
            departure_date = flight_info.get("departure_date")
            return_date = flight_info.get("return_date")
            
            # 필수 정보 확인
            if not destination:
                return {
                    "error": "도착지 정보가 필요합니다.",
                    "suggestion": "예: '서울에서 제주도 항공편 알려줘'"
                }
            
            if not departure_date:
                return {
                    "error": "출발 날짜 정보가 필요합니다.", 
                    "suggestion": "예: '2024-01-15 제주도 항공편'"
                }
            
            # 기본값 설정
            if not origin:
                origin = "서울"  # 기본 출발지
            
            # API 호출
            result = self.client.search_flights_by_route(
                origin_name=origin,
                destination_name=destination,
                departure_date=departure_date,
                return_date=return_date
            )
            
            if isinstance(result, FlightSearchError):
                return {
                    "error": result.error_message,
                    "detail": result.detail
                }
            
            # 검색 결과 포맷팅
            return self._format_flight_results(result)
            
        except Exception as e:
            return {
                "error": f"항공편 검색 중 오류가 발생했습니다: {str(e)}",
                "suggestion": "잠시 후 다시 시도해주세요."
            }
    
    def _format_flight_results(self, response: FlightSearchResponse) -> Dict[str, Any]:
        """검색 결과 포맷팅"""
        if not response.offers:
            return {
                "message": "해당 조건에 맞는 항공편을 찾을 수 없습니다.",
                "suggestion": "날짜나 출발지를 변경해서 다시 검색해보세요."
            }
        
        # 최대 3개 항공편만 표시
        top_offers = response.offers[:3]
        
        # 가격 중복 체크
        unique_prices = set()
        for offer in top_offers:
            unique_prices.add(offer.price.total_amount)
        
        flight_info = {
            "search_summary": {
                "route": f"{response.search_request.origin} → {response.search_request.destination}",
                "departure_date": response.search_request.departure_date.isoformat(),
                "return_date": response.search_request.return_date.isoformat() if response.search_request.return_date else None,
                "total_found": response.total_count,
                "api_status": "⚠️ 테스트 API 사용 중"
            },
            "flights": []
        }
        
        # 모든 가격이 동일한 경우 안내 메시지 추가
        if len(unique_prices) == 1:
            flight_info["price_notice"] = {
                "message": "🤔 모든 항공편 가격이 동일한 이유",
                "reasons": [
                    "• Amadeus Test API의 샘플 데이터 사용",
                    "• 동일한 항공사의 같은 요금 등급 (Economy Y)",
                    "• 국내선 기본 요금 정책의 특성"
                ],
                "suggestion": "실제 예약 시에는 항공사 홈페이지나 여행사에서 정확한 가격을 확인하세요."
            }
        
        for i, offer in enumerate(top_offers, 1):
            flight_data = {
                "rank": i,
                "airline": offer.outbound.segments[0].airline_name,
                "flight_number": offer.outbound.segments[0].flight_number,
                "departure_time": offer.outbound.departure_time.strftime("%H:%M"),
                "arrival_time": offer.outbound.arrival_time.strftime("%H:%M"),
                "duration": offer.outbound.total_duration,
                "price": str(offer.price),
                "is_direct": offer.outbound.is_direct,
                "seats_available": offer.seats_available > 0
            }
            
            # 왕복의 경우 복항 정보 추가
            if offer.inbound:
                flight_data["return_flight"] = {
                    "departure_time": offer.inbound.departure_time.strftime("%H:%M"),
                    "arrival_time": offer.inbound.arrival_time.strftime("%H:%M"),
                    "duration": offer.inbound.total_duration
                }
            
            flight_info["flights"].append(flight_data)
        
        # 최저가 정보
        if response.lowest_price_offer:
            flight_info["lowest_price"] = {
                "amount": str(response.lowest_price_offer.price),
                "airline": response.lowest_price_offer.outbound.segments[0].airline_name
            }
        
        return flight_info
    
    def generate_flight_summary(self, flight_data: Dict[str, Any]) -> str:
        """항공편 정보 요약 생성"""
        if "error" in flight_data:
            return f"❌ {flight_data['error']}\n💡 {flight_data.get('suggestion', '')}"
        
        if "message" in flight_data:
            return f"📢 {flight_data['message']}\n💡 {flight_data.get('suggestion', '')}"
        
        summary = []
        search_info = flight_data.get("search_summary", {})
        
        # 검색 조건 요약
        summary.append(f"✈️ **항공편 검색 결과**")
        summary.append(f"📍 **경로:** {search_info.get('route', '')}")
        summary.append(f"📅 **출발일:** {search_info.get('departure_date', '')}")
        
        if search_info.get('return_date'):
            summary.append(f"🔄 **복항일:** {search_info.get('return_date')}")
        
        summary.append(f"🔍 **총 {search_info.get('total_found', 0)}개 항공편 발견**")
        summary.append("")
        
        # 상위 항공편 정보
        flights = flight_data.get("flights", [])
        for flight in flights:
            summary.append(f"**{flight['rank']}. {flight['airline']} {flight['flight_number']}**")
            summary.append(f"⏰ {flight['departure_time']} → {flight['arrival_time']} ({flight['duration']})")
            summary.append(f"💰 **{flight['price']}**")
            
            if flight['is_direct']:
                summary.append("✅ 직항")
            else:
                summary.append("🔄 경유")
            
            if flight.get('return_flight'):
                ret = flight['return_flight']
                summary.append(f"🔙 복항: {ret['departure_time']} → {ret['arrival_time']} ({ret['duration']})")
            
            summary.append("")
        
        # 최저가 정보
        if flight_data.get("lowest_price"):
            lowest = flight_data["lowest_price"]
            summary.append(f"💸 **최저가:** {lowest['amount']} ({lowest['airline']})")
        
        # 가격 안내 메시지 추가
        if flight_data.get("price_notice"):
            notice = flight_data["price_notice"]
            summary.append("")
            summary.append(f"**{notice['message']}**")
            for reason in notice['reasons']:
                summary.append(reason)
            summary.append(f"💡 {notice['suggestion']}")
        
        return "\n".join(summary)

# 전역 에이전트 인스턴스
flight_agent = FlightSearchAgent() 