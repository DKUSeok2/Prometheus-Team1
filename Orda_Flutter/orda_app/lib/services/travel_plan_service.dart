import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/travel_plan.dart';

class TravelPlanService {
  static const String _kakaoApiKey = '29b88d36376b65d78516805b0ee8ff3b'; // 실제 카카오 REST API 키

  // 실제 제주도 유명 장소들로 구성된 여행 일정
  static String getSampleTravelPlan() {
    return '''
안녕하세요! 3박 4일 제주도 여행 일정을 추천해드릴게요! 🌴

📅 **1일차 - 제주시 중심 탐방**
• 아침 (09:00): 제주국제공항 도착 ✈️
• 점심 (12:00): 명진전복에서 전복 요리 맛보기 🍽️
• 오후 (14:00): 제주의 상징 용두암 관광 🗿
• 오후 (16:00): 동문시장에서 제주 전통시장 구경 🛍️
• 저녁 (18:00): 흑돼지 거리에서 제주 흑돼지 구이 🍖
• 숙박: 제주칼호텔 🏨

📅 **2일차 - 동쪽 해안 코스**
• 아침 (08:00): 성산일출봉 등반 (유네스코 세계자연유산) 🌅
• 오전 (10:30): 일출랜드 관광 🌊
• 점심 (12:30): 해녀의집에서 신선한 해산물 🦀
• 오후 (15:00): 섭지코지 해안 절경 감상 🏖️
• 저녁 (17:30): 서귀포매일올레시장 구경 🛒
• 숙박: 신라호텔제주 🏨

📅 **3일차 - 서쪽 자연 탐방**
• 아침 (09:00): 한라산 어리목탐방로 등반 ⛰️
• 점심 (13:00): 한라산 1100고지 휴게소 🍛
• 오후 (15:30): 중문 천제연폭포 관광 💧
• 오후 (17:00): 중문관광단지 해수욕장 산책 🏖️
• 저녁 (18:30): 돌하르방공원에서 제주 돌하르방 구경 🗿
• 숙박: 롯데호텔제주 🏨

📅 **4일차 - 마무리**
• 아침 (09:30): 중문 주상절리대 관광 🌊
• 오전 (11:00): 테디베어뮤지엄 관람 🧸
• 점심 (13:00): 제주신라면세점에서 쇼핑 🛍️
• 오후 (15:30): 제주국제공항에서 출발 ✈️

즐거운 제주 여행 되세요! 🎉
''';
  }

  // 텍스트에서 여행 일정 파싱
  static List<TravelPlan> parseTravelPlan(String response) {
    List<TravelPlan> plans = [];
    
    try {
      // 일차별로 분할하는 정규식
      RegExp dayPattern = RegExp(r'(\d+일차)[^\n]*\n((?:(?!\d+일차)(?:.*\n?))*)', multiLine: true);
      
      Iterable<RegExpMatch> dayMatches = dayPattern.allMatches(response);
      
      for (RegExpMatch dayMatch in dayMatches) {
        String day = dayMatch.group(1)!; // "1일차"
        String content = dayMatch.group(2)!; // 해당 일차 내용
        
        List<Place> places = _extractPlacesFromContent(content, day);
        
        if (places.isNotEmpty) {
          plans.add(TravelPlan(day: day, places: places));
        }
      }
      
      // 파싱 결과가 없으면 샘플 데이터 사용
      if (plans.isEmpty) {
        return _getSampleTravelPlans();
      }
      
    } catch (e) {
      print('파싱 오류: $e');
      return _getSampleTravelPlans();
    }
    
    return plans;
  }

  // 텍스트 내용에서 장소 추출
  static List<Place> _extractPlacesFromContent(String content, String day) {
    List<Place> places = [];
    int sequence = 1;
    
    // 시간과 장소를 추출하는 정규식
    RegExp placePattern = RegExp(
      r'•?\s*(?:아침|점심|저녁|오전|오후|숙박)?\s*\((\d{2}:\d{2})\):\s*([^🍜🗿🍖🏨🌅🦀🌊🛍️⛰️🍛💧🍽️🛒🍴✈️\n]+)(?:[🍜🗿🍖🏨🌅🦀🌊🛍️⛰️🍛💧🍽️🛒🍴✈️]|$)', 
      multiLine: true
    );
    
    Iterable<RegExpMatch> matches = placePattern.allMatches(content);
    
    for (RegExpMatch match in matches) {
      String time = match.group(1)!; // "09:00"
      String description = match.group(2)!.trim(); // "제주공항 도착"
      
      // 장소명 추출 (더 정교한 방식)
      String placeName = _extractPlaceName(description);
      String category = _categorizePlace(description);
      
      if (placeName.isNotEmpty) {
        places.add(Place(
          name: placeName,
          category: category,
          time: time,
          description: description,
          sequence: sequence++,
        ));
      }
    }
    
    return places;
  }

  // 설명에서 장소명 추출
  static String _extractPlaceName(String description) {
    // 키워드 기반 장소명 추출
    Map<RegExp, String> placePatterns = {
      RegExp(r'(올레국수[^에서\s]*)'): r'\1',
      RegExp(r'(돈사돈)'): r'\1',
      RegExp(r'(라마다플라자 제주호텔)'): r'\1',
      RegExp(r'(성산일출봉)'): r'\1',
      RegExp(r'(성산포횟집)'): r'\1',
      RegExp(r'(섭지코지)'): r'\1',
      RegExp(r'(매일올레시장)'): r'\1',
      RegExp(r'(켄싱턴호텔)'): r'\1',
      RegExp(r'(한라산)'): r'\1',
      RegExp(r'(1100고지 휴게소)'): r'\1',
      RegExp(r'(천제연폭포)'): r'\1',
      RegExp(r'(해신 레스토랑)'): r'\1',
      RegExp(r'(롯데호텔제주)'): r'\1',
      RegExp(r'(이마트 제주점)'): r'\1',
      RegExp(r'(제주공항)'): r'\1',
      RegExp(r'(용두암)'): r'\1',
    };
    
    for (RegExp pattern in placePatterns.keys) {
      RegExpMatch? match = pattern.firstMatch(description);
      if (match != null) {
        return match.group(1)!;
      }
    }
    
    // 일반적인 패턴으로 추출 시도
    RegExp generalPattern = RegExp(r'([가-힣\s]+(?:호텔|펜션|식당|카페|공원|봉|암|포|시장|마트|공항))');
    RegExpMatch? match = generalPattern.firstMatch(description);
    if (match != null) {
      return match.group(1)!.trim();
    }
    
    return '';
  }

  // 장소 카테고리 분류
  static String _categorizePlace(String description) {
    if (description.contains(RegExp(r'호텔|펜션|숙박|체크아웃|조식'))) return '숙박';
    if (description.contains(RegExp(r'국수|식사|맛집|레스토랑|먹기|횟집|음식|저녁식사|점심|저녁|조식'))) return '음식';
    if (description.contains(RegExp(r'일출봉|폭포|공원|등반|구경|산책|탐방|관광'))) return '관광지';
    if (description.contains(RegExp(r'쇼핑|시장|마트|기념품'))) return '쇼핑';
    if (description.contains(RegExp(r'공항|도착|출발'))) return '교통';
    return '기타';
  }

  // 샘플 여행 일정 (파싱 실패 시 사용)
  static List<TravelPlan> _getSampleTravelPlans() {
    return [
      TravelPlan(day: '1일차', places: [
        Place(name: '제주공항', category: '교통', time: '09:00', description: '제주도 도착', sequence: 1),
        Place(name: '올레국수 본점', category: '음식', time: '12:00', description: '고기국수 맛보기', sequence: 2),
        Place(name: '용두암', category: '관광지', time: '14:00', description: '용두암 구경하기', sequence: 3),
        Place(name: '돈사돈', category: '음식', time: '18:00', description: '흑돼지 저녁식사', sequence: 4),
        Place(name: '라마다플라자 제주호텔', category: '숙박', time: '20:00', description: '숙박', sequence: 5),
      ]),
      TravelPlan(day: '2일차', places: [
        Place(name: '성산일출봉', category: '관광지', time: '10:00', description: '성산일출봉 등반', sequence: 1),
        Place(name: '성산포횟집', category: '음식', time: '13:00', description: '해산물 점심', sequence: 2),
        Place(name: '섭지코지', category: '관광지', time: '15:00', description: '섭지코지 산책', sequence: 3),
        Place(name: '매일올레시장', category: '쇼핑', time: '19:00', description: '시장 구경', sequence: 4),
        Place(name: '켄싱턴호텔', category: '숙박', time: '21:00', description: '숙박', sequence: 5),
      ]),
      TravelPlan(day: '3일차', places: [
        Place(name: '한라산', category: '관광지', time: '10:30', description: '한라산 등반', sequence: 1),
        Place(name: '1100고지 휴게소', category: '음식', time: '14:00', description: '간단 식사', sequence: 2),
        Place(name: '천제연폭포', category: '관광지', time: '16:00', description: '폭포 구경', sequence: 3),
        Place(name: '해신 레스토랑', category: '음식', time: '18:30', description: '저녁식사', sequence: 4),
        Place(name: '롯데호텔제주', category: '숙박', time: '20:30', description: '숙박', sequence: 5),
      ]),
    ];
  }

  // 카카오 장소 검색 API
  static Future<void> searchPlaceCoordinates(List<TravelPlan> travelPlans) async {
    for (TravelPlan plan in travelPlans) {
      for (Place place in plan.places) {
        try {
          Map<String, double>? coords = await _searchPlace(place.name);
          if (coords != null) {
            place.latitude = coords['latitude']!;
            place.longitude = coords['longitude']!;
            print('📍 ${place.name}: ${place.latitude}, ${place.longitude}');
          } else {
            // 기본 제주도 좌표 설정
            place.latitude = 33.3617;
            place.longitude = 126.5292;
            print('⚠️ ${place.name}: 좌표를 찾지 못해 기본 좌표 사용');
          }
          
          // API 호출 제한을 위한 딜레이
          await Future.delayed(const Duration(milliseconds: 100));
        } catch (e) {
          print('❌ ${place.name} 검색 오류: $e');
          place.latitude = 33.3617;
          place.longitude = 126.5292;
        }
      }
    }
  }

  // 카카오 장소 검색
  static Future<Map<String, double>?> _searchPlace(String placeName) async {
    try {
      final response = await http.get(
        Uri.parse('https://dapi.kakao.com/v2/local/search/keyword.json?query=$placeName 제주'),
        headers: {'Authorization': 'KakaoAK $_kakaoApiKey'},
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['documents'].isNotEmpty) {
          final place = data['documents'][0];
          return {
            'latitude': double.parse(place['y']),
            'longitude': double.parse(place['x']),
          };
        }
      }
    } catch (e) {
      print('카카오 API 호출 오류: $e');
    }
    return null;
  }

  // 미리 정의된 좌표 (API 키가 없을 때 사용)
  static void setDefaultCoordinates(List<TravelPlan> travelPlans) {
    Map<String, Map<String, double>> defaultCoords = {
      '제주공항': {'latitude': 33.5067, 'longitude': 126.4927},
      '올레국수 본점': {'latitude': 33.5185, 'longitude': 126.5312},
      '용두암': {'latitude': 33.5152, 'longitude': 126.5189},
      '돈사돈': {'latitude': 33.5095, 'longitude': 126.5217},
      '라마다플라자 제주호텔': {'latitude': 33.4996, 'longitude': 126.5311},
      '성산일출봉': {'latitude': 33.4584, 'longitude': 126.9427},
      '성산포횟집': {'latitude': 33.4612, 'longitude': 126.9286},
      '섭지코지': {'latitude': 33.4241, 'longitude': 126.9308},
      '매일올레시장': {'latitude': 33.2484, 'longitude': 126.5614},
      '켄싱턴호텔': {'latitude': 33.2441, 'longitude': 126.5617},
      '한라산': {'latitude': 33.3616, 'longitude': 126.5292},
      '1100고지 휴게소': {'latitude': 33.3822, 'longitude': 126.4953},
      '천제연폭포': {'latitude': 33.2544, 'longitude': 126.4201},
      '해신 레스토랑': {'latitude': 33.2456, 'longitude': 126.4103},
      '롯데호텔제주': {'latitude': 33.2456, 'longitude': 126.4103},
      '이마트 제주점': {'latitude': 33.4887, 'longitude': 126.4982},
    };

    for (TravelPlan plan in travelPlans) {
      for (Place place in plan.places) {
        if (defaultCoords.containsKey(place.name)) {
          place.latitude = defaultCoords[place.name]!['latitude']!;
          place.longitude = defaultCoords[place.name]!['longitude']!;
        } else {
          // 기본 제주도 좌표
          place.latitude = 33.3617;
          place.longitude = 126.5292;
        }
      }
    }
  }
} 