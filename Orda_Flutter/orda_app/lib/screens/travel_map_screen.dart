import 'package:flutter/material.dart';
import 'package:kakao_map_plugin/kakao_map_plugin.dart';
import 'dart:math';
import '../models/travel_plan.dart';
import '../services/travel_plan_service.dart';

class TravelMapScreen extends StatefulWidget {
  final List<TravelPlan> travelPlans;

  const TravelMapScreen({
    super.key,
    required this.travelPlans,
  });

  @override
  State<TravelMapScreen> createState() => _TravelMapScreenState();
}

class _TravelMapScreenState extends State<TravelMapScreen> {
  late KakaoMapController mapController;
  Set<Polyline> polylines = {};
  int selectedDayIndex = -1; // -1: 전체보기, 0~: 특정 일차
  Map<String, double> placeDistances = {}; // 장소 간 거리 저장
  LatLng? mapCenter; // 계산된 지도 중심점

  @override
  void initState() {
    super.initState();
    _debugTravelPlans(); // 디버깅 추가
    _initializeMarkers();
    _calculateDistances();
  }

  // 여행 일정 디버깅
  void _debugTravelPlans() {
    print('\n🔍 === 여행 일정 디버깅 ===');
    print('총 일차 수: ${widget.travelPlans.length}');
    
    for (int i = 0; i < widget.travelPlans.length; i++) {
      final plan = widget.travelPlans[i];
      print('\n📅 ${plan.day}: ${plan.places.length}개 장소');
      
      for (int j = 0; j < plan.places.length; j++) {
        final place = plan.places[j];
        print('  ${j+1}. ${place.name} (${place.category}) - ${place.time}');
        print('     좌표: ${place.latitude}, ${place.longitude}');
      }
    }
    print('========================\n');
  }

  // 폴리라인 초기화 (마커 제거로 오류 방지)
  void _initializeMarkers() {
    polylines.clear();

    // 지도 중심점 계산
    _calculateMapCenter();

    for (int dayIndex = 0; dayIndex < widget.travelPlans.length; dayIndex++) {
      final dayPlan = widget.travelPlans[dayIndex];
      final dayColor = _getDayColor(dayIndex);
      
      // 특정 일차 선택 시 해당 일차만 표시
      if (selectedDayIndex != -1 && selectedDayIndex != dayIndex) {
        continue;
      }

      for (int placeIndex = 0; placeIndex < dayPlan.places.length; placeIndex++) {
        final place = dayPlan.places[placeIndex];
        
        if (place.latitude != 0.0 && place.longitude != 0.0) {
          // 마커 추가 안함 - JavaScript 오류 완전 방지
          String categoryIcon = TravelConstants.categoryIcons[place.category] ?? '📍';
          print('📍 장소 확인: ${place.name} ($categoryIcon, ${dayPlan.day}): ${place.latitude}, ${place.longitude}');

          // 같은 일차 내 다음 장소와 경로 연결
          if (placeIndex < dayPlan.places.length - 1) {
            final nextPlace = dayPlan.places[placeIndex + 1];
            if (nextPlace.latitude != 0.0 && nextPlace.longitude != 0.0) {
              // 거리 계산
              double distance = _calculateDistance(
                place.latitude, place.longitude,
                nextPlace.latitude, nextPlace.longitude,
              );
              
              String routeKey = '${place.name}_${nextPlace.name}';
              placeDistances[routeKey] = distance;
              
              polylines.add(
                Polyline(
                  polylineId: '${dayPlan.day}_route_$placeIndex',
                  points: [
                    LatLng(place.latitude, place.longitude),
                    LatLng(nextPlace.latitude, nextPlace.longitude),
                  ],
                  strokeColor: dayColor,
                  strokeWidth: 3,
                  strokeOpacity: 0.8,
                      ),
    );
  }

  // 거리 정보 표시
  Widget _buildDistanceInfo() {
    Map<String, double> dayDistances = {};
    double totalDistance = 0;
    
    // 일차별 거리 계산
    for (int dayIndex = 0; dayIndex < widget.travelPlans.length; dayIndex++) {
      final dayPlan = widget.travelPlans[dayIndex];
      double dayDistance = 0;
      
      // 특정 일차 선택 시 해당 일차만 계산
      if (selectedDayIndex != -1 && selectedDayIndex != dayIndex) {
        continue;
      }
      
      for (int i = 0; i < dayPlan.places.length - 1; i++) {
        final place1 = dayPlan.places[i];
        final place2 = dayPlan.places[i + 1];
        
        if (place1.latitude != 0.0 && place1.longitude != 0.0 && 
            place2.latitude != 0.0 && place2.longitude != 0.0) {
          
          double distance = _calculateDistance(
            place1.latitude, place1.longitude,
            place2.latitude, place2.longitude,
          );
          
          dayDistance += distance;
        }
      }
      
      if (dayDistance > 0) {
        dayDistances[dayPlan.day] = dayDistance;
        totalDistance += dayDistance;
      }
    }
    
    if (dayDistances.isEmpty) {
      return Container();
    }
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.straighten, size: 16, color: Colors.blue),
              const SizedBox(width: 6),
              const Text(
                '이동 거리',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // 일차별 거리
          ...dayDistances.entries.map((entry) {
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    entry.key,
                    style: const TextStyle(fontSize: 11),
                  ),
                  Text(
                    '${entry.value.toStringAsFixed(1)}km',
                    style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
          
          // 구분선
          if (dayDistances.length > 1) ...[
            const Divider(height: 16, thickness: 1),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '총 거리',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  '${totalDistance.toStringAsFixed(1)}km',
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}
        }
      }
    }
    
    setState(() {});
  }

  // 일차별 색상 가져오기
  Color _getDayColor(int dayIndex) {
    String colorHex = TravelConstants.dayColors[dayIndex % TravelConstants.dayColors.length];
    return Color(int.parse(colorHex.substring(1, 7), radix: 16) + 0xFF000000);
  }

  // 마커 탭 시 정보 표시
  void _showMarkerInfo(String markerId) {
    try {
      // markerId 파싱: "1일차_1" -> day="1일차", sequence=1
      List<String> parts = markerId.split('_');
      if (parts.length != 2) return;
      
      String day = parts[0];
      int sequence = int.parse(parts[1]);
      
      // 해당 장소 찾기
      Place? targetPlace;
      TravelPlan? targetPlan;
      
      for (TravelPlan plan in widget.travelPlans) {
        if (plan.day == day) {
          targetPlan = plan;
          for (Place place in plan.places) {
            if (place.sequence == sequence) {
              targetPlace = place;
              break;
            }
          }
          break;
        }
      }
      
      if (targetPlace != null && targetPlan != null) {
        // Non-null assertion을 위한 지역 변수
        final place = targetPlace!;
        final plan = targetPlan!;
        final planIndex = widget.travelPlans.indexOf(plan);
        
        String categoryIcon = TravelConstants.categoryIcons[place.category] ?? '📍';
        
        showModalBottomSheet(
          context: context,
          backgroundColor: Colors.transparent,
          builder: (context) => Container(
            margin: const EdgeInsets.all(16),
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    // 큰 카테고리 아이콘 (대체 솔루션)
                    Container(
                      width: 50,
                      height: 50,
                      decoration: BoxDecoration(
                        color: _getDayColor(planIndex).withOpacity(0.1),
                        borderRadius: BorderRadius.circular(25),
                        border: Border.all(
                          color: _getDayColor(planIndex),
                          width: 2,
                        ),
                      ),
                      child: Center(
                        child: Text(
                          categoryIcon,
                          style: const TextStyle(fontSize: 28),
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            place.name,
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: _getDayColor(planIndex),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              '${plan.day} • ${place.category}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    const Icon(Icons.access_time, size: 16, color: Colors.grey),
                    const SizedBox(width: 4),
                    Text(
                      place.time,
                      style: const TextStyle(color: Colors.grey),
                    ),
                    const SizedBox(width: 16),
                    const Icon(Icons.star, size: 16, color: Colors.amber),
                    const SizedBox(width: 4),
                    Text(
                      '제주 명소',
                      style: const TextStyle(color: Colors.grey),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  place.description,
                  style: const TextStyle(
                    fontSize: 14,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () => Navigator.pop(context),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _getDayColor(planIndex),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: const Text(
                      '닫기',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      }
    } catch (e) {
      print('마커 정보 표시 오류: $e');
    }
  }



  // 지도 중심점 계산
  void _calculateMapCenter() {
    List<LatLng> allPoints = [];
    
    for (TravelPlan plan in widget.travelPlans) {
      for (Place place in plan.places) {
        if (place.latitude != 0.0 && place.longitude != 0.0) {
          allPoints.add(LatLng(place.latitude, place.longitude));
        }
      }
    }
    
    if (allPoints.isNotEmpty) {
      double centerLat = allPoints.map((p) => p.latitude).reduce((a, b) => a + b) / allPoints.length;
      double centerLng = allPoints.map((p) => p.longitude).reduce((a, b) => a + b) / allPoints.length;
      mapCenter = LatLng(centerLat, centerLng);
    } else {
      // 기본값: 제주공항 중심
      mapCenter = LatLng(33.5097, 126.4929);
    }
  }

  // 두 좌표 간 거리 계산 (하버사인 공식, km 단위)
  double _calculateDistance(double lat1, double lon1, double lat2, double lon2) {
    const double earthRadius = 6371; // 지구 반지름 (km)
    
    double dLat = _degreesToRadians(lat2 - lat1);
    double dLon = _degreesToRadians(lon2 - lon1);
    
    double a = 
        (sin(dLat / 2) * sin(dLat / 2)) +
        cos(_degreesToRadians(lat1)) * cos(_degreesToRadians(lat2)) *
        (sin(dLon / 2) * sin(dLon / 2));
    
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    
    return earthRadius * c;
  }
  
  double _degreesToRadians(double degrees) {
    return degrees * (pi / 180);
  }

  // 거리 정보 계산 및 출력
  void _calculateDistances() {
    print('\n🗺️ === 여행 일정 거리 정보 ===');
    
    double totalDistance = 0;
    
    for (int dayIndex = 0; dayIndex < widget.travelPlans.length; dayIndex++) {
      final dayPlan = widget.travelPlans[dayIndex];
      double dayDistance = 0;
      
      print('\n📅 ${dayPlan.day}:');
      
      for (int i = 0; i < dayPlan.places.length - 1; i++) {
        final place1 = dayPlan.places[i];
        final place2 = dayPlan.places[i + 1];
        
        if (place1.latitude != 0.0 && place1.longitude != 0.0 && 
            place2.latitude != 0.0 && place2.longitude != 0.0) {
          
          double distance = _calculateDistance(
            place1.latitude, place1.longitude,
            place2.latitude, place2.longitude,
          );
          
          dayDistance += distance;
          totalDistance += distance;
          
          print('  ${place1.name} → ${place2.name}: ${distance.toStringAsFixed(1)}km');
        }
      }
      
      print('  ${dayPlan.day} 총 거리: ${dayDistance.toStringAsFixed(1)}km');
    }
    
    print('\n🎯 전체 여행 총 거리: ${totalDistance.toStringAsFixed(1)}km');
    print('=========================\n');
  }

  // 줌인 기능 (지도 중심으로 확대)
  void _zoomIn() {
    if (mapCenter != null) {
      // 더 정확한 중심점으로 이동 (확대 효과)
      _moveToDayCenter(selectedDayIndex);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('더 자세한 보기는 지도를 직접 조작해주세요'),
          duration: Duration(seconds: 1),
          backgroundColor: Colors.blue,
        ),
      );
    }
  }

  // 줌아웃 기능 (전체 보기)
  void _zoomOut() {
    // 전체 여행 일정이 보이도록 중심 이동
    _moveToDayCenter(-1);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('전체 일정 보기로 이동'),
        duration: Duration(seconds: 1),
        backgroundColor: Colors.green,
      ),
    );
  }

  // 지도 중심 이동
  void _moveToDayCenter(int dayIndex) {
    if (dayIndex >= 0 && dayIndex < widget.travelPlans.length) {
      final dayPlaces = widget.travelPlans[dayIndex].places
          .where((place) => place.latitude != 0.0 && place.longitude != 0.0)
          .toList();

      if (dayPlaces.isNotEmpty) {
        // 해당 일차 장소들의 중심점 계산
        double centerLat = dayPlaces.map((p) => p.latitude).reduce((a, b) => a + b) / dayPlaces.length;
        double centerLng = dayPlaces.map((p) => p.longitude).reduce((a, b) => a + b) / dayPlaces.length;
        
        mapController.setCenter(LatLng(centerLat, centerLng));
      }
    } else {
      // 전체 중심으로 이동 (계산된 중심점 사용)
      if (mapCenter != null) {
        mapController.setCenter(mapCenter!);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('여행 일정 지도'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 1,
        actions: [
          // 전체보기 버튼
          IconButton(
            icon: Icon(
              Icons.map,
              color: selectedDayIndex == -1 ? const Color(0xFF6B73FF) : Colors.grey,
            ),
            onPressed: () {
              setState(() {
                selectedDayIndex = -1;
              });
              _initializeMarkers();
              _moveToDayCenter(-1);
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          // 카카오 지도 (경로선만 표시 - 오류 방지)
          KakaoMap(
            onMapCreated: (controller) async {
              mapController = controller;
              // 계산된 중심점으로 이동
              if (mapCenter != null) {
                await controller.setCenter(mapCenter!);
              }
            },
            polylines: polylines.toList(),
            center: mapCenter ?? LatLng(33.5097, 126.4929), // 제주공항 중심점 사용
          ),

          // 일차별 필터 버튼들
          Positioned(
            top: 20,
            left: 20,
            right: 20,
            child: _buildDayFilterButtons(),
          ),

          // 범례
          Positioned(
            bottom: 180,
            left: 20,
            child: _buildLegend(),
          ),

          // 장소 목록 카드 (마커 대체)
          Positioned(
            bottom: 100,
            left: 20,
            right: 20,
            child: _buildPlaceListCard(),
          ),

          // 거리 정보
          Positioned(
            bottom: 230,
            left: 20,
            right: 20,
            child: _buildDistanceInfo(),
          ),

          // 줌 컨트롤 버튼들
          Positioned(
            right: 20,
            bottom: 20,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // 선택 일차 중심으로 이동
                Container(
                  width: 50,
                  height: 50,
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(25),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.2),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: IconButton(
                    onPressed: () {
                      _zoomIn();
                    },
                    icon: const Icon(
                      Icons.center_focus_strong,
                      color: Color(0xFF6B73FF),
                      size: 24,
                    ),
                  ),
                ),
                
                // 전체 일정 보기
                Container(
                  width: 50,
                  height: 50,
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(25),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.2),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: IconButton(
                    onPressed: () {
                      _zoomOut();
                    },
                    icon: const Icon(
                      Icons.fullscreen,
                      color: Color(0xFF6B73FF),
                      size: 24,
                    ),
                  ),
                ),
                
                // 현재 위치/중심 이동 버튼
                Container(
                  width: 50,
                  height: 50,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(25),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.2),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: IconButton(
                    onPressed: () {
                      _moveToDayCenter(selectedDayIndex);
                    },
                    icon: const Icon(
                      Icons.my_location,
                      color: Color(0xFF6B73FF),
                      size: 24,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // 일차별 필터 버튼
  Widget _buildDayFilterButtons() {
    return Container(
      height: 50,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: widget.travelPlans.length + 1, // +1 for 전체보기
        itemBuilder: (context, index) {
          bool isSelected = (index == 0) ? selectedDayIndex == -1 : selectedDayIndex == index - 1;
          String label = (index == 0) ? '전체' : widget.travelPlans[index - 1].day;
          Color buttonColor = (index == 0) 
              ? const Color(0xFF6B73FF) 
              : _getDayColor(index - 1);

          return GestureDetector(
            onTap: () {
              setState(() {
                selectedDayIndex = (index == 0) ? -1 : index - 1;
              });
              _initializeMarkers();
              _moveToDayCenter(selectedDayIndex);
            },
            child: Container(
              margin: const EdgeInsets.only(right: 10),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: isSelected ? buttonColor : Colors.white,
                borderRadius: BorderRadius.circular(25),
                border: Border.all(color: buttonColor, width: 2),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Center(
                child: Text(
                  label,
                  style: TextStyle(
                    color: isSelected ? Colors.white : buttonColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // 범례
  Widget _buildLegend() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '범례',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 8),
          ...TravelConstants.categoryIcons.entries.map((entry) {
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(entry.value, style: const TextStyle(fontSize: 14)),
                  const SizedBox(width: 6),
                  Text(
                    entry.key,
                    style: const TextStyle(fontSize: 10),
                  ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }

  // 장소 정보 표시
  void _showPlaceInfo(String markerId) {
    // markerId에서 일차와 시퀀스 추출 (예: "1일차_1")
    List<String> parts = markerId.split('_');
    if (parts.length >= 2) {
      String day = parts[0];
      int sequence = int.tryParse(parts[1]) ?? 1;

      // 해당 장소 찾기
      Place? targetPlace;
      for (TravelPlan plan in widget.travelPlans) {
        if (plan.day == day) {
          for (Place place in plan.places) {
            if (place.sequence == sequence) {
              targetPlace = place;
              break;
            }
          }
          break;
        }
      }

      if (targetPlace != null) {
        showModalBottomSheet(
          context: context,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          builder: (context) => _buildPlaceInfoSheet(targetPlace!, day),
        );
      }
    }
  }

  // 장소 정보 시트
  Widget _buildPlaceInfoSheet(Place place, String day) {
    // 다음 장소까지의 거리 찾기
    String? nextPlaceDistance;
    
    for (TravelPlan plan in widget.travelPlans) {
      if (plan.day == day) {
        for (int i = 0; i < plan.places.length - 1; i++) {
          if (plan.places[i].sequence == place.sequence) {
            Place nextPlace = plan.places[i + 1];
            double distance = _calculateDistance(
              place.latitude, place.longitude,
              nextPlace.latitude, nextPlace.longitude,
            );
            nextPlaceDistance = '다음 목적지(${nextPlace.name})까지 ${distance.toStringAsFixed(1)}km';
            break;
          }
        }
        break;
      }
    }
    
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                TravelConstants.categoryIcons[place.category] ?? '📍',
                style: const TextStyle(fontSize: 24),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      place.name,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      '$day • ${place.time} • ${place.category}',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            place.description,
            style: const TextStyle(fontSize: 16),
          ),
          
          // 거리 정보 표시
          if (nextPlaceDistance != null) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  const Icon(Icons.directions_car, size: 16, color: Colors.blue),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      nextPlaceDistance!,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.blue,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
          
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF6B73FF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: const Text('확인'),
            ),
          ),
        ],
      ),
    );
  }

  // 거리 정보 표시
  Widget _buildDistanceInfo() {
    Map<String, double> dayDistances = {};
    double totalDistance = 0;
    
    // 일차별 거리 계산
    for (int dayIndex = 0; dayIndex < widget.travelPlans.length; dayIndex++) {
      final dayPlan = widget.travelPlans[dayIndex];
      double dayDistance = 0;
      
      // 특정 일차 선택 시 해당 일차만 계산
      if (selectedDayIndex != -1 && selectedDayIndex != dayIndex) {
        continue;
      }
      
      for (int i = 0; i < dayPlan.places.length - 1; i++) {
        final place1 = dayPlan.places[i];
        final place2 = dayPlan.places[i + 1];
        
        if (place1.latitude != 0.0 && place1.longitude != 0.0 && 
            place2.latitude != 0.0 && place2.longitude != 0.0) {
          
          double distance = _calculateDistance(
            place1.latitude, place1.longitude,
            place2.latitude, place2.longitude,
          );
          
          dayDistance += distance;
        }
      }
      
      if (dayDistance > 0) {
        dayDistances[dayPlan.day] = dayDistance;
        totalDistance += dayDistance;
      }
    }
    
    if (dayDistances.isEmpty) {
      return Container();
    }
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.straighten, size: 16, color: Colors.blue),
              const SizedBox(width: 6),
              const Text(
                '이동 거리',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // 일차별 거리
          ...dayDistances.entries.map((entry) {
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    entry.key,
                    style: const TextStyle(fontSize: 11),
                  ),
                  Text(
                    '${entry.value.toStringAsFixed(1)}km',
                    style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
          
          // 구분선
          if (dayDistances.length > 1) ...[
            const Divider(height: 16, thickness: 1),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '총 거리',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  '${totalDistance.toStringAsFixed(1)}km',
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  // 장소 목록 카드 (마커 대체)
  Widget _buildPlaceListCard() {
    if (selectedDayIndex == -1) {
      // 전체 보기일 때는 간단한 요약만
      int totalPlaces = 0;
      for (TravelPlan plan in widget.travelPlans) {
        totalPlaces += plan.places.length;
      }
      
      return Container(
        constraints: const BoxConstraints(
          maxHeight: 75,
          minHeight: 50,
        ),
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '🗺️ 제주도 3박 4일 여행',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              '총 ${widget.travelPlans.length}일차 • $totalPlaces개 장소',
              style: const TextStyle(color: Colors.grey, fontSize: 10),
            ),
            const SizedBox(height: 2),
            const Text(
              '🔴 1일차  🔵 2일차  🟢 3일차  🟠 4일차',
              style: TextStyle(fontSize: 9),
            ),
          ],
        ),
      );
    } else {
      // 특정 일차 선택 시 해당 장소들 표시
      TravelPlan selectedPlan = widget.travelPlans[selectedDayIndex];
      
      return Container(
        constraints: const BoxConstraints(
          maxHeight: 75,
          minHeight: 50,
        ),
        padding: const EdgeInsets.all(6),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _getDayColor(selectedDayIndex),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '${selectedPlan.day} 장소 목록',
                  style: const TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 2),
            Expanded(
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: selectedPlan.places.length,
                itemBuilder: (context, index) {
                  Place place = selectedPlan.places[index];
                  String categoryIcon = TravelConstants.categoryIcons[place.category] ?? '📍';
                  
                  return Container(
                    width: 70,
                    margin: const EdgeInsets.only(right: 4),
                    padding: const EdgeInsets.all(2),
                    decoration: BoxDecoration(
                      color: _getDayColor(selectedDayIndex).withOpacity(0.1),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                        color: _getDayColor(selectedDayIndex).withOpacity(0.3),
                      ),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          categoryIcon,
                          style: const TextStyle(fontSize: 12),
                        ),
                        const SizedBox(height: 1),
                        Text(
                          place.name,
                          style: const TextStyle(
                            fontSize: 7,
                            fontWeight: FontWeight.w500,
                          ),
                          textAlign: TextAlign.center,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        Text(
                          place.time,
                          style: const TextStyle(
                            fontSize: 6,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      );
    }
  }
} 