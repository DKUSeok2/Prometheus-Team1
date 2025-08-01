import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'auth_service.dart';

class ChatService {
  static const String _baseUrl = 'http://192.168.219.106:8000/api/v1';
  static const String _currentSessionKey = 'current_session_id';
  final AuthService _authService = AuthService();

  // 현재 시간 가져오기
  static String getCurrentTime() {
    final now = DateTime.now();
    return '${now.hour}:${now.minute.toString().padLeft(2, '0')}';
  }

  // 메시지 모델
  static Map<String, dynamic> createMessage(String text, bool isUser) {
    return {
      'text': text,
      'isUser': isUser,
      'time': getCurrentTime(),
      'timestamp': DateTime.now().toIso8601String(),
    };
  }

  // 챗봇에게 메시지 전송
  Future<Map<String, dynamic>> sendMessage(String message, {
    String? sessionId,
    String? userName,
    String? personality,
    String? travelStyle,
  }) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.post(
        Uri.parse('$_baseUrl/chat'),
        headers: headers,
        body: json.encode({
          'content': message,
          if (sessionId != null) 'session_id': sessionId,
          if (userName != null) 'user_name': userName,
          if (personality != null) 'personality': personality,
          if (travelStyle != null) 'travel_style': travelStyle,
        }),
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        // 세션 ID 저장
        await _saveCurrentSessionId(data['session_id']);
        
        return {
          'success': true,
          'response': data['response'],
          'session_id': data['session_id'],
          'needs_more_info': data['needs_more_info'],
          'profile_completion': data['profile_completion'],
          'follow_up_questions': data['follow_up_questions'],
          'user_profile': data['user_profile'],
          'analysis_confidence': data['analysis_confidence'],
        };
      } else {
        return {'success': false, 'error': data['detail'] ?? '메시지 전송 실패'};
      }
    } catch (e) {
      print('메시지 전송 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 채팅 히스토리 조회
  Future<Map<String, dynamic>> getChatHistory(String sessionId) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.get(
        Uri.parse('$_baseUrl/chat/history/$sessionId'),
        headers: headers,
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        // Flutter UI를 위한 메시지 포맷 변환
        List<Map<String, dynamic>> messages = [];
        if (data['messages'] != null) {
          for (var msg in data['messages']) {
            messages.add({
              'text': msg['content'],
              'isUser': msg['message_type'] == 'user',
              'time': _formatTime(msg['created_at']),
              'timestamp': msg['created_at'],
              'metadata': msg['message_metadata'],
              'has_itinerary': _isItineraryResponse(msg['content']),
            });
          }
        }
        
        return {
          'success': true,
          'messages': messages,
          'session_info': {
            'session_id': data['session_id'],
            'user_profile': data['user_profile'],
            'profile_completion': data['profile_completion'],
            'created_at': data['created_at'],
          }
        };
      } else {
        return {'success': false, 'error': data['detail'] ?? '히스토리 조회 실패'};
      }
    } catch (e) {
      print('히스토리 조회 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 사용자 프로필 조회
  Future<Map<String, dynamic>> getUserProfile(String sessionId) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.get(
        Uri.parse('$_baseUrl/chat/profile/$sessionId'),
        headers: headers,
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {
          'success': true,
          'profile': data['profile'],
          'completion': data['completion'],
        };
      } else {
        return {'success': false, 'error': data['detail'] ?? '프로필 조회 실패'};
      }
    } catch (e) {
      print('프로필 조회 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 사용자 프로필 업데이트
  Future<Map<String, dynamic>> updateUserProfile(
    String sessionId,
    Map<String, dynamic> profileUpdates,
  ) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.put(
        Uri.parse('$_baseUrl/chat/profile/$sessionId'),
        headers: headers,
        body: json.encode(profileUpdates),
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {
          'success': true,
          'profile': data['profile'],
          'completion': data['completion'],
        };
      } else {
        return {'success': false, 'error': data['detail'] ?? '프로필 업데이트 실패'};
      }
    } catch (e) {
      print('프로필 업데이트 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 세션 리셋
  Future<Map<String, dynamic>> resetSession(String sessionId) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.delete(
        Uri.parse('$_baseUrl/chat/session/$sessionId'),
        headers: headers,
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        // 로컬 세션 ID도 초기화
        await _clearCurrentSessionId();
        return {'success': true, 'message': data['message']};
      } else {
        return {'success': false, 'error': data['detail'] ?? '세션 리셋 실패'};
      }
    } catch (e) {
      print('세션 리셋 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 현재 세션 ID 가져오기
  Future<String?> getCurrentSessionId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_currentSessionKey);
  }

  // 현재 세션 ID 저장
  Future<void> _saveCurrentSessionId(String sessionId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_currentSessionKey, sessionId);
  }

  // 현재 세션 ID 초기화
  Future<void> _clearCurrentSessionId() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_currentSessionKey);
  }

  // 새로운 채팅 세션 시작
  Future<String?> startNewSession() async {
    await _clearCurrentSessionId();
    return null; // 새로운 세션은 첫 메시지 전송 시 자동 생성됨
  }

  // 모든 채팅 세션 목록 가져오기
  Future<Map<String, dynamic>> getAllChatSessions() async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.get(
        Uri.parse('$_baseUrl/chat/sessions'),
        headers: headers,
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {
          'success': true,
          'sessions': data['sessions'] ?? [],
        };
      } else {
        return {
          'success': false,
          'error': data['detail'] ?? '세션 목록을 불러올 수 없습니다',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': e.toString(),
      };
    }
  }

  // 채팅 세션 제목 업데이트
  Future<Map<String, dynamic>> updateChatSessionTitle(String sessionId, String newTitle) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.put(
        Uri.parse('$_baseUrl/chat/sessions/$sessionId/title'),
        headers: headers,
        body: json.encode({'title': newTitle}),
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {'success': true, 'message': data['message']};
      } else {
        return {'success': false, 'error': data['detail'] ?? '제목 변경에 실패했습니다'};
      }
    } catch (e) {
      return {'success': false, 'error': e.toString()};
    }
  }

  // 시간 포맷팅 헬퍼
  String _formatTime(String isoString) {
    try {
      final dateTime = DateTime.parse(isoString);
      return '${dateTime.hour}:${dateTime.minute.toString().padLeft(2, '0')}';
    } catch (e) {
      return getCurrentTime();
    }
  }

  // 일정/여행계획이 포함된 응답인지 판단하는 로직
  bool _isItineraryResponse(String response) {
    final keywords = ['숙박 추천', '액티비티 추천', '### ', '1. **', '2. **', '일정', '여행', '추천'];
    return keywords.any((keyword) => response.contains(keyword));
  }

  // 대화형 프로필 정보 추출 헬퍼
  Map<String, String> getProfileSuggestions() {
    return {
      'companions': '누구와 함께 여행하시나요? (가족/친구/연인/혼자)',
      'duration': '여행 기간은 어느 정도 생각하고 계시나요? (당일/1박2일/2박3일/3박4일/장기)',
      'budget': '예산은 어느 정도로 생각하고 계시나요? (절약/보통/여유/럭셔리)',
      'interests': '어떤 활동에 관심이 있으시나요? (음식/바다/산/관광지/문화/액티비티)',
      'transportation': '이동 수단은 어떻게 생각하고 계시나요? (렌터카/대중교통/택시/도보)',
      'accommodation_preference': '어떤 숙박시설을 선호하시나요? (호텔/리조트/펜션/게스트하우스/민박)',
    };
  }

  // 채팅방 제목 변경
  Future<Map<String, dynamic>> updateChatRoomTitle(String sessionId, String newTitle) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.put(
        Uri.parse('$_baseUrl/chat/sessions/$sessionId/title'),
        headers: headers,
        body: json.encode({
          'title': newTitle,
        }),
      );

      if (response.statusCode == 200) {
        return {
          'success': true,
          'message': '채팅방 제목이 변경되었습니다.',
        };
      } else {
        final data = json.decode(response.body);
        return {
          'success': false,
          'error': data['message'] ?? '제목 변경에 실패했습니다.',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': '네트워크 오류: $e',
      };
    }
  }

  // 채팅방 삭제
  Future<Map<String, dynamic>> deleteChatRoom(String sessionId) async {
    try {
      final headers = await _authService.getAuthHeaders();
      
      final response = await http.delete(
        Uri.parse('$_baseUrl/chat/session/$sessionId'),
        headers: headers,
      );

      if (response.statusCode == 200) {
        return {
          'success': true,
          'message': '채팅방이 삭제되었습니다.',
        };
      } else {
        final data = json.decode(response.body);
        return {
          'success': false,
          'error': data['message'] ?? '채팅방 삭제에 실패했습니다.',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': '네트워크 오류: $e',
      };
    }
  }
} 