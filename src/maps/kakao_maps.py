"""
카카오 지도 API 연동 모듈
"""
import requests
import folium
import streamlit.components.v1 as components
from typing import Dict, Any, List, Optional, Tuple
import json
from ..config import settings

class KakaoMapsClient:
    """카카오 지도 API 클라이언트"""
    
    def __init__(self):
        self.api_key = settings.kakao_api_key
        self.base_url = "https://dapi.kakao.com/v2/local"
        self.headers = {
            "Authorization": f"KakaoAK {self.api_key}"
        } if self.api_key else {}
    
    def search_address(self, address: str) -> Optional[Dict[str, Any]]:
        """주소로 좌표 검색"""
        if not self.api_key:
            print("❌ 카카오 API 키가 설정되지 않았습니다.")
            return None
        
        url = f"{self.base_url}/search/address.json"
        params = {"query": address}
        
        print(f"🔍 주소 검색 시작: '{address}'")
        print(f"📍 API URL: {url}")
        print(f"🔑 API 키: {self.api_key[:10]}..." if self.api_key else "❌ API 키 없음")
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            print(f"📡 HTTP 응답 코드: {response.status_code}")
            
            if response.status_code == 401:
                print("❌ 인증 실패: API 키를 확인해주세요.")
                return None
            elif response.status_code == 403:
                print("❌ 권한 거부: API 키 권한을 확인해주세요.")
                return None
            
            response.raise_for_status()
            
            data = response.json()
            print(f"📋 API 응답: {data}")
            
            if data.get("documents"):
                result = data["documents"][0]
                coordinates = {
                    "address": result.get("address_name", ""),
                    "road_address": result.get("road_address_name", ""),
                    "latitude": float(result.get("y", 0)),
                    "longitude": float(result.get("x", 0))
                }
                print(f"✅ 좌표 변환 성공: {coordinates}")
                return coordinates
            else:
                print(f"⚠️ 주소를 찾을 수 없습니다: '{address}'")
                
                # 제주도 주소가 아닌 경우 제주도를 추가해서 재시도
                if "제주" not in address:
                    retry_address = f"제주도 {address}"
                    print(f"🔄 제주도 주소로 재시도: '{retry_address}'")
                    return self.search_address(retry_address)
                
                return None
                
        except requests.Timeout:
            print("⏰ 요청 시간 초과: 네트워크 연결을 확인해주세요.")
        except requests.ConnectionError:
            print("🌐 연결 오류: 인터넷 연결을 확인해주세요.")
        except requests.RequestException as e:
            print(f"📡 HTTP 요청 오류: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"📋 오류 상세: {error_data}")
                except:
                    print(f"📋 응답 내용: {e.response.text}")
        except (ValueError, KeyError) as e:
            print(f"🔧 응답 파싱 오류: {e}")
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
        
        return None
    
    def search_keyword(self, keyword: str, location: Tuple[float, float] = None, radius: int = 10000) -> List[Dict[str, Any]]:
        """키워드로 장소 검색"""
        if not self.api_key:
            print("카카오 API 키가 설정되지 않았습니다.")
            return []
        
        url = f"{self.base_url}/search/keyword.json"
        params = {
            "query": keyword,
            "size": 15
        }
        
        if location:
            params.update({
                "x": location[1],  # longitude
                "y": location[0],  # latitude
                "radius": radius
            })
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for doc in data.get("documents", []):
                results.append({
                    "name": doc.get("place_name", ""),
                    "address": doc.get("address_name", ""),
                    "road_address": doc.get("road_address_name", ""),
                    "phone": doc.get("phone", ""),
                    "category": doc.get("category_name", ""),
                    "latitude": float(doc.get("y", 0)),
                    "longitude": float(doc.get("x", 0)),
                    "place_url": doc.get("place_url", ""),
                    "distance": doc.get("distance", "")
                })
            
            return results
            
        except requests.RequestException as e:
            print(f"키워드 검색 오류: {e}")
        except (ValueError, KeyError) as e:
            print(f"응답 파싱 오류: {e}")
        
        return []
    
    def get_directions(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """길찾기 API (카카오 네비)"""
        if not self.api_key:
            print("카카오 API 키가 설정되지 않았습니다.")
            return None
        
        url = "https://apis-navi.kakaomobility.com/v1/directions"
        params = {
            "origin": f"{origin[1]},{origin[0]}",  # longitude, latitude
            "destination": f"{destination[1]},{destination[0]}",
            "waypoints": "",
            "priority": "RECOMMEND",
            "car_fuel": "GASOLINE",
            "car_hipass": "false",
            "alternatives": "false",
            "road_details": "false"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get("routes"):
                route = data["routes"][0]
                summary = route.get("summary", {})
                
                return {
                    "distance": summary.get("distance", 0),  # 미터
                    "duration": summary.get("duration", 0),  # 초
                    "taxi_fare": summary.get("fare", {}).get("taxi", 0),
                    "toll_fare": summary.get("fare", {}).get("toll", 0),
                    "waypoints": route.get("sections", [])
                }
        except requests.RequestException as e:
            print(f"길찾기 오류: {e}")
        except (ValueError, KeyError) as e:
            print(f"응답 파싱 오류: {e}")
        
        return None

class MapVisualizer:
    """카카오 지도 시각화 클래스"""
    
    def __init__(self):
        self.kakao_client = KakaoMapsClient()
        # 제주도 중심 좌표
        self.jeju_center = (33.4996, 126.5312)
    
    def create_kakao_map_html(self, center: Tuple[float, float] = None, markers: List[Dict[str, Any]] = None, zoom: int = 10) -> str:
        """카카오 지도 HTML 생성"""
        if center is None:
            center = self.jeju_center
        
        if markers is None:
            markers = []
        
        # 마커 데이터를 JSON으로 변환
        markers_json = json.dumps(markers, ensure_ascii=False)
        
        # 카카오 JavaScript API 키
        app_key = settings.kakao_js_key or settings.kakao_api_key  # JavaScript 키 우선, 없으면 REST API 키
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>카카오 지도</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Malgun Gothic', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }}
                #map {{
                    width: 100%;
                    height: 500px;
                    border-radius: 8px;
                    border: 2px solid #ddd;
                }}
                .loading {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 500px;
                    background: #f8f9fa;
                    color: #6c757d;
                    font-size: 14px;
                    border-radius: 8px;
                    border: 2px solid #ddd;
                }}
                .info-window {{
                    padding: 10px;
                    font-family: 'Malgun Gothic', sans-serif;
                    font-size: 12px;
                    width: 200px;
                }}
                .info-title {{
                    font-weight: bold;
                    font-size: 14px;
                    margin-bottom: 5px;
                    color: #333;
                }}
                .info-content {{
                    color: #666;
                    line-height: 1.4;
                }}
            </style>
        </head>
        <body>
            <div id="map" class="loading">🗺️ 지도를 불러오는 중...</div>
            
            <script>
                // 카카오 지도 SDK 동적 로드
                function loadKakaoMapSDK() {{
                    return new Promise((resolve, reject) => {{
                        // 이미 로드되었는지 확인
                        if (window.kakao && window.kakao.maps) {{
                            resolve();
                            return;
                        }}
                        
                                                 const script = document.createElement('script');
                         script.type = 'text/javascript';
                         script.src = '//dapi.kakao.com/v2/maps/sdk.js?appkey={app_key}&libraries=services';
                        script.onload = () => {{
                            // 카카오 지도 SDK 초기화 대기
                            kakao.maps.load(() => {{
                                resolve();
                            }});
                        }};
                        script.onerror = (error) => {{
                            reject(new Error('카카오 지도 SDK 로드 실패'));
                        }};
                        document.head.appendChild(script);
                    }});
                }}
                
                                // 지도 초기화
                async function initializeMap() {{
                    try {{
                        // SDK 로드 대기
                        await loadKakaoMapSDK();
                        
                        var mapContainer = document.getElementById('map');
                        mapContainer.className = ''; // loading 클래스 제거
                        
                        var mapOption = {{
                            center: new kakao.maps.LatLng({center[0]}, {center[1]}),
                            level: {zoom}
                        }};
                        
                        var map = new kakao.maps.Map(mapContainer, mapOption);
                    
                        // 마커 데이터
                        var markers = {markers_json};
                        
                        // 카테고리별 마커 이미지
                        var markerImages = {{
                            '음식': 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png',
                            '숙소': 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/marker_red.png',
                            '관광지': 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/marker_blue.png',
                            '행사': 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/marker_gold.png'
                        }};
                        
                        // 마커와 인포윈도우 배열
                        var kakaoMarkers = [];
                        var infoWindows = [];
                        
                        // 마커 생성
                        markers.forEach(function(markerData, index) {{
                            var markerPosition = new kakao.maps.LatLng(markerData.lat, markerData.lng);
                            
                            // 마커 이미지 설정
                            var imageSrc = markerImages[markerData.category] || 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/marker_red.png';
                            var imageSize = new kakao.maps.Size(24, 35);
                            var markerImage = new kakao.maps.MarkerImage(imageSrc, imageSize);
                            
                            // 마커 생성
                            var marker = new kakao.maps.Marker({{
                                position: markerPosition,
                                image: markerImage
                            }});
                            
                            marker.setMap(map);
                            kakaoMarkers.push(marker);
                            
                            // 인포윈도우 내용
                            var infoContent = 
                                '<div class="info-window">' +
                                '<div class="info-title">' + markerData.name + '</div>' +
                                '<div class="info-content">' +
                                '<strong>카테고리:</strong> ' + markerData.category + '<br>' +
                                '<strong>주소:</strong> ' + markerData.address + '<br>' +
                                (markerData.phone ? '<strong>전화:</strong> ' + markerData.phone + '<br>' : '') +
                                '<strong>설명:</strong> ' + markerData.description.substring(0, 100) + '...' +
                                '</div>' +
                                '</div>';
                            
                            var infoWindow = new kakao.maps.InfoWindow({{
                                content: infoContent
                            }});
                            
                            infoWindows.push(infoWindow);
                            
                            // 마커 클릭 이벤트
                            kakao.maps.event.addListener(marker, 'click', function() {{
                                // 다른 인포윈도우 닫기
                                infoWindows.forEach(function(iw) {{
                                    iw.close();
                                }});
                                
                                // 클릭한 마커의 인포윈도우 열기
                                infoWindow.open(map, marker);
                            }});
                        }});
                        
                        // 마커들이 모두 보이도록 지도 범위 조정
                        if (markers.length > 0) {{
                            var bounds = new kakao.maps.LatLngBounds();
                            markers.forEach(function(markerData) {{
                                bounds.extend(new kakao.maps.LatLng(markerData.lat, markerData.lng));
                            }});
                            map.setBounds(bounds);
                        }}
                        
                        // 지도 타입 컨트롤 추가
                        var mapTypeControl = new kakao.maps.MapTypeControl();
                        map.addControl(mapTypeControl, kakao.maps.ControlPosition.TOPRIGHT);
                        
                        // 줌 컨트롤 추가
                        var zoomControl = new kakao.maps.ZoomControl();
                        map.addControl(zoomControl, kakao.maps.ControlPosition.RIGHT);
                        
                    }} catch (error) {{
                        console.error('카카오 지도 로드 실패:', error);
                        document.getElementById('map').innerHTML = 
                            '<div class="loading">⚠️ 지도를 불러올 수 없습니다<br><small>' + error.message + '</small></div>';
                    }}
                }}
                
                // 페이지 로드 후 지도 초기화
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initializeMap);
                }} else {{
                    initializeMap();
                }}
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def create_map(self, center: Tuple[float, float] = None, zoom: int = 10) -> str:
        """기본 카카오 지도 생성"""
        return self.create_kakao_map_html(center, [], zoom)
    
    def create_travel_route_map(self, recommendations: List[Dict[str, Any]]) -> str:
        """여행 경로 카카오 지도 생성"""
        if not recommendations:
            return self.create_map()
        
        # 추천 장소들을 마커 데이터로 변환
        markers = []
        
        for rec in recommendations:
            # 주소로 좌표 검색
            coords = self._get_coordinates(rec.get("address", ""))
            
            if coords:
                markers.append({
                    "lat": coords[0],
                    "lng": coords[1],
                    "name": rec.get("name", "이름 없음"),
                    "category": rec.get("category", "기타"),
                    "address": rec.get("address", "주소 없음"),
                    "phone": rec.get("phone", ""),
                    "description": rec.get("description", "설명 없음")
                })
        
        # 중심점 계산
        if markers:
            center_lat = sum(m["lat"] for m in markers) / len(markers)
            center_lng = sum(m["lng"] for m in markers) / len(markers)
            center = (center_lat, center_lng)
        else:
            center = self.jeju_center
        
        return self.create_kakao_map_html(center, markers, zoom=11)
    
    def _get_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """주소로 좌표 검색"""
        if not address:
            return None
        
        result = self.kakao_client.search_address(address)
        if result:
            return (result["latitude"], result["longitude"])
        
        return None
    
    def calculate_travel_time_matrix(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """여행지간 시간/거리 매트릭스 계산"""
        if len(locations) < 2:
            return {"error": "최소 2개 이상의 장소가 필요합니다."}
        
        # 각 장소의 좌표 얻기
        coordinates = []
        location_names = []
        
        for location in locations:
            coords = self._get_coordinates(location.get("address", ""))
            if coords:
                coordinates.append(coords)
                location_names.append(location.get("name", "이름 없음"))
        
        if len(coordinates) < 2:
            return {"error": "좌표를 찾을 수 없는 장소가 있습니다."}
        
        # 거리/시간 매트릭스 계산
        matrix = []
        for i, origin in enumerate(coordinates):
            row = []
            for j, destination in enumerate(coordinates):
                if i == j:
                    row.append({"distance": 0, "duration": 0})
                else:
                    route_info = self.kakao_client.get_directions(origin, destination)
                    if route_info:
                        row.append({
                            "distance": route_info["distance"],
                            "duration": route_info["duration"]
                        })
                    else:
                        # 직선거리로 대체 계산
                        import math
                        lat1, lon1 = origin
                        lat2, lon2 = destination
                        
                        # Haversine 공식으로 직선거리 계산
                        R = 6371000  # 지구 반지름 (미터)
                        lat1_rad = math.radians(lat1)
                        lat2_rad = math.radians(lat2)
                        delta_lat = math.radians(lat2 - lat1)
                        delta_lon = math.radians(lon2 - lon1)
                        
                        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
                             math.cos(lat1_rad) * math.cos(lat2_rad) *
                             math.sin(delta_lon/2) * math.sin(delta_lon/2))
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        distance = R * c
                        
                        # 예상 소요시간 (평균 속도 40km/h 가정)
                        duration = distance / (40 * 1000 / 3600)  # 초 단위
                        
                        row.append({
                            "distance": int(distance),
                            "duration": int(duration)
                        })
            matrix.append(row)
        
        return {
            "locations": location_names,
            "matrix": matrix
        }

# 전역 인스턴스
kakao_client = KakaoMapsClient()
map_visualizer = MapVisualizer()
