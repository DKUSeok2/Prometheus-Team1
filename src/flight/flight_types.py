"""
항공권 검색 관련 데이터 타입 정의
"""
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class CabinClass(Enum):
    """항공편 좌석 등급"""
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"

class TripType(Enum):
    """여행 유형"""
    ONE_WAY = "one-way"
    ROUND_TRIP = "round-trip"

@dataclass
class FlightSearchRequest:
    """항공편 검색 요청"""
    origin: str  # 출발지 공항 코드 (예: ICN)
    destination: str  # 도착지 공항 코드 (예: CJU)
    departure_date: date  # 출발 날짜
    return_date: Optional[date] = None  # 복항 날짜 (왕복의 경우)
    adults: int = 1  # 성인 승객 수
    children: int = 0  # 아동 승객 수
    infants: int = 0  # 유아 승객 수
    cabin_class: CabinClass = CabinClass.ECONOMY  # 좌석 등급
    max_results: int = 10  # 최대 결과 수
    currency: str = "KRW"  # 통화
    
    @property
    def trip_type(self) -> TripType:
        """여행 유형 자동 감지"""
        return TripType.ROUND_TRIP if self.return_date else TripType.ONE_WAY

@dataclass
class FlightSegment:
    """항공편 구간 정보"""
    departure_airport: str  # 출발 공항 코드
    arrival_airport: str  # 도착 공항 코드
    departure_time: datetime  # 출발 시간
    arrival_time: datetime  # 도착 시간
    airline_code: str  # 항공사 코드
    airline_name: str  # 항공사 이름
    flight_number: str  # 항공편 번호
    aircraft_type: str  # 항공기 기종
    duration: str  # 비행 시간
    cabin_class: str  # 좌석 등급

@dataclass
class FlightItinerary:
    """항공편 일정"""
    segments: List[FlightSegment]  # 구간 목록
    total_duration: str  # 총 여행 시간
    
    @property
    def departure_time(self) -> datetime:
        """출발 시간"""
        return self.segments[0].departure_time
    
    @property
    def arrival_time(self) -> datetime:
        """도착 시간"""
        return self.segments[-1].arrival_time
    
    @property
    def is_direct(self) -> bool:
        """직항 여부"""
        return len(self.segments) == 1

@dataclass
class FlightPrice:
    """항공편 가격 정보"""
    total_amount: float  # 총 금액
    base_amount: float  # 기본 요금
    taxes_amount: float  # 세금 및 수수료
    currency: str  # 통화
    
    def __str__(self) -> str:
        """가격 문자열 표현"""
        return f"{self.total_amount:,.0f} {self.currency}"

@dataclass
class FlightOffer:
    """항공편 제안"""
    id: str  # 제안 ID
    outbound: FlightItinerary  # 가는 편
    inbound: Optional[FlightItinerary]  # 오는 편 (왕복의 경우)
    price: FlightPrice  # 가격 정보
    seats_available: int  # 잔여 좌석 수
    instant_ticketing_required: bool  # 즉시 발권 필요 여부
    last_ticketing_date: Optional[date]  # 마지막 발권 날짜
    
    @property
    def is_round_trip(self) -> bool:
        """왕복 여부"""
        return self.inbound is not None
    
    @property
    def total_duration(self) -> str:
        """총 여행 시간"""
        if self.is_round_trip:
            return f"가는편: {self.outbound.total_duration}, 오는편: {self.inbound.total_duration}"
        return self.outbound.total_duration

@dataclass
class FlightSearchResponse:
    """항공편 검색 응답"""
    offers: List[FlightOffer]  # 항공편 제안 목록
    search_request: FlightSearchRequest  # 검색 요청 정보
    total_count: int  # 전체 결과 수
    
    @property
    def lowest_price_offer(self) -> Optional[FlightOffer]:
        """최저가 항공편"""
        if not self.offers:
            return None
        return min(self.offers, key=lambda x: x.price.total_amount)
    
    @property
    def direct_flights(self) -> List[FlightOffer]:
        """직항 항공편들"""
        return [offer for offer in self.offers if offer.outbound.is_direct]

@dataclass
class FlightSearchError:
    """항공편 검색 오류"""
    error_code: str  # 오류 코드
    error_message: str  # 오류 메시지
    detail: Optional[str] = None  # 상세 정보

# 공항 코드 매핑
AIRPORT_CODES = {
    # 한국 주요 공항
    "인천": "ICN",
    "인천국제공항": "ICN", 
    "서울": "ICN",
    "김포": "GMP",
    "김포공항": "GMP",
    "부산": "PUS",
    "김해": "PUS",
    "김해공항": "PUS",
    "제주": "CJU", 
    "제주도": "CJU",
    "제주국제공항": "CJU",
    "대구": "TAE",
    "대구공항": "TAE",
    "광주": "KWJ",
    "광주공항": "KWJ",
    
    # 해외 주요 공항
    "도쿄": "NRT",
    "나리타": "NRT",
    "하네다": "HND",
    "오사카": "KIX",
    "간사이": "KIX",
    "후쿠오카": "FUK",
    "상하이": "PVG",
    "푸동": "PVG",
    "베이징": "PEK",
    "홍콩": "HKG",
    "타이베이": "TPE",
    "방콕": "BKK",
    "싱가포르": "SIN",
}

def get_airport_code(location_name: str) -> Optional[str]:
    """지역명으로 공항 코드 찾기"""
    return AIRPORT_CODES.get(location_name.strip())

def format_duration(duration_iso: str) -> str:
    """ISO 8601 duration을 읽기 쉬운 형태로 변환"""
    # PT1H30M -> 1시간 30분
    import re
    
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?', duration_iso)
    if not match:
        return duration_iso
    
    hours, minutes = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    
    if hours and minutes:
        return f"{hours}시간 {minutes}분"
    elif hours:
        return f"{hours}시간"
    elif minutes:
        return f"{minutes}분"
    else:
        return "정보 없음" 