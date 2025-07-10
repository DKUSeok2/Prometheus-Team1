"""
Amadeus API 클라이언트
"""
import requests
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Union
from ..config.settings import settings
from .flight_types import (
    FlightSearchRequest, FlightSearchResponse, FlightOffer, 
    FlightItinerary, FlightSegment, FlightPrice, FlightSearchError,
    format_duration, get_airport_code
)

class AmadeusClient:
    """Amadeus API 클라이언트"""
    
    def __init__(self):
        self.client_id = settings.amadeus_client_id
        self.client_secret = settings.amadeus_client_secret  
        self.base_url = settings.amadeus_base_url
        self._access_token = None
        self._token_expires_at = None
        
        # 항공사 코드 매핑
        self.airline_names = {
            "KE": "대한항공",
            "OZ": "아시아나항공", 
            "LJ": "진에어",
            "BX": "에어부산",
            "RS": "에어서울",
            "TW": "티웨이항공",
            "ZE": "이스타항공",
            "4V": "플라이강원",
            "LQ": "란코스타일항공",
            "JL": "일본항공",
            "NH": "전일본공수",
            "LCC": "Peach",
            "MM": "Peach",
            "GK": "젯스타 재팬",
        }
    
    def _get_access_token(self) -> str:
        """OAuth2 액세스 토큰 획득"""
        # 토큰이 유효하면 재사용
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at:
                return self._access_token
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Amadeus API 클라이언트 ID와 시크릿이 설정되지 않았습니다.")
        
        url = f"{self.base_url}/v1/security/oauth2/token"
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data["access_token"]
            
            # 토큰 만료 시간 설정 (약간의 여유를 둠)
            expires_in = token_data.get("expires_in", 1799)  # 기본 30분
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
            
            return self._access_token
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Amadeus 토큰 획득 실패: {str(e)}")
    
    def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """API 요청 수행"""
        token = self._get_access_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=params)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error_description', str(e))
                except:
                    error_msg = str(e)
            else:
                error_msg = str(e)
            
            raise Exception(f"Amadeus API 요청 실패: {error_msg}")
    
    def search_flights(self, request: FlightSearchRequest) -> Union[FlightSearchResponse, FlightSearchError]:
        """항공편 검색"""
        try:
            # API 요청 파라미터 구성
            params = {
                "originLocationCode": request.origin,
                "destinationLocationCode": request.destination,
                "departureDate": request.departure_date.isoformat(),
                "adults": request.adults,
                "max": min(request.max_results, 10),  # 더 많은 결과 요청
                "currencyCode": request.currency,
                "travelClass": request.cabin_class.value
            }
            
            # 왕복의 경우 복항 날짜 추가
            if request.return_date:
                params["returnDate"] = request.return_date.isoformat()
            
            # 아동, 유아 승객이 있는 경우 추가
            if request.children > 0:
                params["children"] = request.children
            if request.infants > 0:
                params["infants"] = request.infants
            
            # API 호출
            response_data = self._make_request("GET", "/v2/shopping/flight-offers", params)
            
            # 응답 파싱
            offers = []
            for offer_data in response_data.get("data", []):
                try:
                    offer = self._parse_flight_offer(offer_data)
                    offers.append(offer)
                except Exception as e:
                    print(f"항공편 제안 파싱 오류: {e}")
                    continue
            
            return FlightSearchResponse(
                offers=offers,
                search_request=request,
                total_count=len(offers)
            )
            
        except Exception as e:
            return FlightSearchError(
                error_code="SEARCH_ERROR",
                error_message=str(e),
                detail="항공편 검색 중 오류가 발생했습니다."
            )
    
    def _parse_flight_offer(self, offer_data: Dict[str, Any]) -> FlightOffer:
        """항공편 제안 데이터 파싱"""
        offer_id = offer_data.get("id", "")
        
        # 가격 정보 파싱
        price_data = offer_data.get("price", {})
        price = FlightPrice(
            total_amount=float(price_data.get("grandTotal", 0)),
            base_amount=float(price_data.get("base", 0)),
            taxes_amount=float(price_data.get("grandTotal", 0)) - float(price_data.get("base", 0)),
            currency=price_data.get("currency", "KRW")
        )
        
        # 일정 정보 파싱
        itineraries_data = offer_data.get("itineraries", [])
        
        # 가는 편 (첫 번째 일정)
        outbound = self._parse_itinerary(itineraries_data[0]) if itineraries_data else None
        
        # 오는 편 (두 번째 일정, 왕복의 경우)
        inbound = None
        if len(itineraries_data) > 1:
            inbound = self._parse_itinerary(itineraries_data[1])
        
        # 기타 정보
        traveler_pricings = offer_data.get("travelerPricings", [])
        seats_available = len(traveler_pricings)  # 간접적으로 추정
        
        return FlightOffer(
            id=offer_id,
            outbound=outbound,
            inbound=inbound,
            price=price,
            seats_available=seats_available,
            instant_ticketing_required=offer_data.get("instantTicketingRequired", False),
            last_ticketing_date=self._parse_date(offer_data.get("lastTicketingDate"))
        )
    
    def _parse_itinerary(self, itinerary_data: Dict[str, Any]) -> FlightItinerary:
        """일정 데이터 파싱"""
        segments_data = itinerary_data.get("segments", [])
        segments = []
        
        for segment_data in segments_data:
            segment = self._parse_segment(segment_data)
            segments.append(segment)
        
        total_duration = format_duration(itinerary_data.get("duration", ""))
        
        return FlightItinerary(
            segments=segments,
            total_duration=total_duration
        )
    
    def _parse_segment(self, segment_data: Dict[str, Any]) -> FlightSegment:
        """구간 데이터 파싱"""
        departure = segment_data.get("departure", {})
        arrival = segment_data.get("arrival", {})
        carrier_code = segment_data.get("carrierCode", "")
        
        return FlightSegment(
            departure_airport=departure.get("iataCode", ""),
            arrival_airport=arrival.get("iataCode", ""),
            departure_time=self._parse_datetime(departure.get("at")),
            arrival_time=self._parse_datetime(arrival.get("at")),
            airline_code=carrier_code,
            airline_name=self.airline_names.get(carrier_code, carrier_code),
            flight_number=f"{carrier_code}{segment_data.get('number', '')}",
            aircraft_type=segment_data.get("aircraft", {}).get("code", ""),
            duration=format_duration(segment_data.get("duration", "")),
            cabin_class=segment_data.get("cabin", "")
        )
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """날짜/시간 문자열 파싱"""
        if not datetime_str:
            return datetime.now()
        
        try:
            # ISO 8601 형식 파싱
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except:
            return datetime.now()
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """날짜 문자열 파싱"""
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str).date()
        except:
            return None
    
    def search_flights_by_route(self, origin_name: str, destination_name: str, 
                               departure_date: str, return_date: str = None) -> Union[FlightSearchResponse, FlightSearchError]:
        """지역명으로 항공편 검색"""
        try:
            from .flight_types import AIRPORT_CODES
            
            # 지역명을 공항 코드로 변환
            origin_code = get_airport_code(origin_name)
            destination_code = get_airport_code(destination_name)
            
            if not origin_code:
                korean_airports = [k for k, v in AIRPORT_CODES.items() if v in ["ICN", "GMP", "PUS", "CJU"]]
                return FlightSearchError(
                    error_code="INVALID_ORIGIN",
                    error_message=f"출발지 '{origin_name}'에 해당하는 공항을 찾을 수 없습니다.",
                    detail="지원하는 출발지: " + ", ".join(korean_airports)
                )
            
            if not destination_code:
                all_locations = list(AIRPORT_CODES.keys())
                return FlightSearchError(
                    error_code="INVALID_DESTINATION", 
                    error_message=f"도착지 '{destination_name}'에 해당하는 공항을 찾을 수 없습니다.",
                    detail="지원하는 도착지: " + ", ".join(all_locations[:10]) + "..."
                )
            
            # 날짜 파싱
            try:
                dep_date = datetime.strptime(departure_date, "%Y-%m-%d").date()
            except:
                return FlightSearchError(
                    error_code="INVALID_DATE",
                    error_message="출발 날짜 형식이 올바르지 않습니다. (YYYY-MM-DD 형식 사용)",
                    detail=f"입력된 날짜: {departure_date}"
                )
            
            ret_date = None
            if return_date:
                try:
                    ret_date = datetime.strptime(return_date, "%Y-%m-%d").date()
                except:
                    return FlightSearchError(
                        error_code="INVALID_RETURN_DATE",
                        error_message="복항 날짜 형식이 올바르지 않습니다. (YYYY-MM-DD 형식 사용)",
                        detail=f"입력된 날짜: {return_date}"
                    )
            
            # 검색 요청 생성
            request = FlightSearchRequest(
                origin=origin_code,
                destination=destination_code,
                departure_date=dep_date,
                return_date=ret_date,
                max_results=5  # 요약 정보용으로 5개만
            )
            
            return self.search_flights(request)
            
        except Exception as e:
            return FlightSearchError(
                error_code="GENERAL_ERROR",
                error_message=str(e),
                detail="항공편 검색 중 예상치 못한 오류가 발생했습니다."
            )
    
    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            token = self._get_access_token()
            return bool(token)
        except:
            return False

# 전역 클라이언트 인스턴스
amadeus_client = AmadeusClient() 