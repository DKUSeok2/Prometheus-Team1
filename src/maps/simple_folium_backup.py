"""
임시 Folium 지도 백업
"""
import folium
from typing import Dict, Any, List, Optional, Tuple

class SimpleMapVisualizer:
    """간단한 Folium 지도 시각화"""
    
    def __init__(self):
        self.jeju_center = (33.4996, 126.5312)
    
    def create_map(self, center: Tuple[float, float] = None, zoom: int = 10) -> folium.Map:
        """기본 지도 생성"""
        if center is None:
            center = self.jeju_center
        
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='CartoDB positron'
        )
        
        return m
    
    def create_travel_route_map(self, recommendations: List[Dict[str, Any]]) -> folium.Map:
        """여행 경로 지도 생성"""
        from ..maps.kakao_maps import KakaoMapsClient
        
        # 기본 지도 생성
        m = self.create_map()
        
        # 카카오 클라이언트로 좌표 검색
        kakao_client = KakaoMapsClient()
        
        colors = {
            "음식": "red",
            "숙소": "blue", 
            "관광지": "green",
            "행사": "purple"
        }
        
        for i, rec in enumerate(recommendations):
            # 주소로 좌표 검색
            coords = self._get_coordinates(rec.get("address", ""), kakao_client)
            
            if coords:
                category = rec.get("category", "기타")
                color = colors.get(category, "gray")
                
                # 팝업 내용 구성
                popup_content = f"""
                <div style="width: 200px;">
                    <h4>{rec.get('name', '이름 없음')}</h4>
                    <p><b>카테고리:</b> {category}</p>
                    <p><b>주소:</b> {rec.get('address', '주소 없음')}</p>
                    {f"<p><b>전화:</b> {rec.get('phone', '')}</p>" if rec.get('phone') else ""}
                    <p><b>설명:</b> {rec.get('description', '')[:100]}...</p>
                </div>
                """
                
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=rec.get('name', '이름 없음'),
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(m)
        
        return m
    
    def _get_coordinates(self, address: str, kakao_client) -> Optional[Tuple[float, float]]:
        """주소로 좌표 검색"""
        if not address:
            return None
        
        result = kakao_client.search_address(address)
        if result:
            return (result["latitude"], result["longitude"])
        
        return None

# 전역 인스턴스
simple_map_visualizer = SimpleMapVisualizer() 