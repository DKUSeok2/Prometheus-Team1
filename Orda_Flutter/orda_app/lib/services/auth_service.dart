import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class AuthService {
  static const String _baseUrl = 'http://192.168.219.106:8000/api/v1'; // FastAPI 백엔드 URL
  static const String _tokenKey = 'auth_token';
  static const String _userKey = 'user_data';

  // 회원가입 메서드
  Future<Map<String, dynamic>> register(String email, String password, String nickname) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/auth/register'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'password': password,
          'nickname': nickname,
        }),
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {'success': true, 'data': data};
      } else {
        return {'success': false, 'error': data['detail'] ?? '회원가입 실패'};
      }
    } catch (e) {
      print('회원가입 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 이메일 중복 확인
  Future<Map<String, dynamic>> checkEmailAvailability(String email) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/auth/check-email?email=$email'),
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {'success': true, 'available': data['available']};
      } else {
        return {'success': false, 'error': data['detail'] ?? '이메일 확인 실패'};
      }
    } catch (e) {
      print('이메일 확인 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 로그인 메서드
  Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'password': password,
        }),
      );

      final data = json.decode(response.body);

      if (response.statusCode == 200) {
        // 토큰 저장
        await _saveToken(data['access_token']);
        
        // 사용자 정보 가져오기 (토큰 검증과 동시에)
        final userResult = await _fetchUserInfo();
        if (userResult['success']) {
          await _saveUserData(userResult['data']);
          return {'success': true, 'data': userResult['data']};
        }
        
        return {'success': true, 'message': '로그인 성공'};
      } else {
        return {'success': false, 'error': data['detail'] ?? '로그인 실패'};
      }
    } catch (e) {
      print('로그인 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 사용자 정보 가져오기
  Future<Map<String, dynamic>> _fetchUserInfo() async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/auth/me'),
        headers: await getAuthHeaders(),
      );

      final data = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return {'success': true, 'data': data};
      } else {
        return {'success': false, 'error': data['detail'] ?? '사용자 정보 조회 실패'};
      }
    } catch (e) {
      print('사용자 정보 조회 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 로그아웃 메서드
  Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_userKey);
  }

  // 로그인 상태 확인
  Future<bool> isLoggedIn() async {
    final token = await getToken();
    if (token == null) return false;
    
    // 토큰 유효성 검사
    return await validateToken();
  }

  // 저장된 토큰 가져오기
  Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey);
  }

  // 저장된 사용자 정보 가져오기
  Future<Map<String, dynamic>?> getUserData() async {
    final prefs = await SharedPreferences.getInstance();
    final userJson = prefs.getString(_userKey);
    if (userJson != null) {
      return json.decode(userJson);
    }
    return null;
  }

  // 토큰 저장
  Future<void> _saveToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token);
  }

  // 사용자 정보 저장
  Future<void> _saveUserData(Map<String, dynamic> userData) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_userKey, json.encode(userData));
  }

  // API 요청 시 헤더에 토큰 포함
  Future<Map<String, String>> getAuthHeaders() async {
    final token = await getToken();
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  // 토큰 유효성 검사
  Future<bool> validateToken() async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/auth/validate'),
        headers: await getAuthHeaders(),
      );

      return response.statusCode == 200;
    } catch (e) {
      print('토큰 검증 오류: $e');
      return false;
    }
  }

  // 비밀번호 변경
  Future<Map<String, dynamic>> changePassword(String currentPassword, String newPassword) async {
    try {
      final response = await http.put(
        Uri.parse('$_baseUrl/auth/change-password'),
        headers: await getAuthHeaders(),
        body: json.encode({
          'current_password': currentPassword,
          'new_password': newPassword,
        }),
      );

      final data = json.decode(response.body);

      if (response.statusCode == 200) {
        return {'success': true, 'message': data['message']};
      } else {
        return {'success': false, 'error': data['detail'] ?? '비밀번호 변경 실패'};
      }
    } catch (e) {
      print('비밀번호 변경 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }

  // 비밀번호 재설정 - 이메일로 임시 비밀번호 전송
  Future<Map<String, dynamic>> resetPassword(String email) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/auth/reset-password'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
        }),
      );

      final data = json.decode(response.body);

      if (response.statusCode == 200) {
        return {
          'success': true, 
          'message': data['message'],
          'email': data['email']
        };
      } else {
        return {'success': false, 'error': data['detail'] ?? '비밀번호 재설정 실패'};
      }
    } catch (e) {
      print('비밀번호 재설정 오류: $e');
      return {'success': false, 'error': '네트워크 오류가 발생했습니다.'};
    }
  }
} 