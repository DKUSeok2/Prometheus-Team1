import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class AuthService {
  static const String _baseUrl = 'https://api.example.com'; // 실제 API URL로 변경 필요
  static const String _tokenKey = 'auth_token';
  static const String _userKey = 'user_data';

  // 로그인 메서드
  Future<bool> login(String email, String password) async {
    try {
      // 데모용 로그인 (실제 API 호출 대신 간단한 검증)
      await Future.delayed(const Duration(seconds: 2)); // 네트워크 요청 시뮬레이션
      
      // 데모용 계정 확인
      if (email == 'demo@orda.com' && password == '123456') {
        // 토큰과 사용자 정보 저장
        await _saveUserData({
          'email': email,
          'name': '데모 사용자',
          'id': '1',
        });
        await _saveToken('demo_token_123456');
        return true;
      }
      
      // 실제 API 호출 코드 (주석 처리)
      /*
      final response = await http.post(
        Uri.parse('$_baseUrl/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'password': password,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        await _saveToken(data['token']);
        await _saveUserData(data['user']);
        return true;
      }
      */
      
      return false;
    } catch (e) {
      print('로그인 오류: $e');
      return false;
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
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString(_tokenKey);
    return token != null && token.isNotEmpty;
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

  // 토큰 유효성 검사 (실제 API 호출)
  Future<bool> validateToken() async {
    try {
      final token = await getToken();
      if (token == null) return false;

      // 데모용 토큰 검증
      if (token == 'demo_token_123456') {
        return true;
      }

      // 실제 API 호출 코드 (주석 처리)
      /*
      final response = await http.get(
        Uri.parse('$_baseUrl/auth/validate'),
        headers: await getAuthHeaders(),
      );

      return response.statusCode == 200;
      */
      
      return false;
    } catch (e) {
      print('토큰 검증 오류: $e');
      return false;
    }
  }
} 