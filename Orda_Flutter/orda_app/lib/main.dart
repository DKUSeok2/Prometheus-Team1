import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:kakao_map_plugin/kakao_map_plugin.dart';
import 'screens/splash_screen.dart';
import 'screens/onboarding_screen.dart';
import 'screens/home_screen.dart';
import 'services/auth_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // .env 파일 로드 (없으면 무시)
  try {
    await dotenv.load(fileName: ".env");
  } catch (e) {
    print('No .env file found, using default values');
  }
  
  // 카카오 지도 초기화 (API 키가 없으면 빈 문자열)
  try {
    AuthRepository.initialize(
      appKey: dotenv.env['KAKAO_MAP_API_KEY'] ?? '', 
    );
  } catch (e) {
    print('Kakao Map initialization failed: $e');
  }
  
  runApp(const OrdaApp());
}

class OrdaApp extends StatefulWidget {
  const OrdaApp({super.key});

  @override
  State<OrdaApp> createState() => _OrdaAppState();
}

class _OrdaAppState extends State<OrdaApp> {
  bool _isLoading = true;
  bool _isLoggedIn = false;

  @override
  void initState() {
    super.initState();
    _checkAuthStatus();
  }

  Future<void> _checkAuthStatus() async {
    // 항상 온보딩 화면부터 시작하도록 설정
    // 기존 로그인 토큰 제거
    final authService = AuthService();
    await authService.logout();
    
    setState(() {
      _isLoggedIn = false;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '오르다',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
        fontFamily: 'NanumGothic', // NanumGothic 폰트 사용
      ),
      home: const SplashScreen(), // 바로 스플래시 화면부터 시작
      debugShowCheckedModeBanner: false,
    );
  }
}
