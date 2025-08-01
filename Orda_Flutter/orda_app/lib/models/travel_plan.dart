class TravelPlan {
  final String day;
  final List<Place> places;

  TravelPlan({
    required this.day,
    required this.places,
  });

  factory TravelPlan.fromJson(Map<String, dynamic> json) {
    return TravelPlan(
      day: json['day'],
      places: (json['places'] as List)
          .map((place) => Place.fromJson(place))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'day': day,
      'places': places.map((place) => place.toJson()).toList(),
    };
  }
}

class Place {
  final String name;
  final String category;
  final String time;
  final String description;
  double latitude;
  double longitude;
  final int sequence;

  Place({
    required this.name,
    required this.category,
    required this.time,
    required this.description,
    this.latitude = 0.0,
    this.longitude = 0.0,
    required this.sequence,
  });

  factory Place.fromJson(Map<String, dynamic> json) {
    return Place(
      name: json['name'],
      category: json['category'],
      time: json['time'],
      description: json['description'],
      latitude: json['latitude']?.toDouble() ?? 0.0,
      longitude: json['longitude']?.toDouble() ?? 0.0,
      sequence: json['sequence'] ?? 0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'category': category,
      'time': time,
      'description': description,
      'latitude': latitude,
      'longitude': longitude,
      'sequence': sequence,
    };
  }
}

// 카테고리별 색상 및 아이콘
class TravelConstants {
  static const Map<String, String> categoryColors = {
    '음식': '#FF6B6B',      // 빨간색
    '숙박': '#4ECDC4',      // 청록색
    '관광지': '#45B7D1',    // 파란색
    '쇼핑': '#FFA726',      // 주황색
    '교통': '#9C27B0',      // 보라색
    '기타': '#78909C',      // 회색
  };

  static const Map<String, String> categoryIcons = {
    '음식': '🍽️',
    '숙박': '🏨',
    '관광지': '🏛️',
    '쇼핑': '🛍️',
    '교통': '✈️',
    '기타': '📍',
  };

  static const List<String> dayColors = [
    '#FF5722', // 1일차 - 빨간색
    '#2196F3', // 2일차 - 파란색
    '#4CAF50', // 3일차 - 초록색
    '#FF9800', // 4일차 - 주황색
    '#9C27B0', // 5일차 - 보라색
    '#607D8B', // 6일차 - 회색
    '#E91E63', // 7일차 - 핑크색
  ];
} 