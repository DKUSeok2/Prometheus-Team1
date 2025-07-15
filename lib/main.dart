import 'package:flutter/material.dart';
import 'screens/onboarding_screen.dart';
import 'screens/home_screen.dart';
import 'services/auth_service.dart';

void main() {
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
        fontFamily: 'NotoSans', // 한국어 폰트 설정
      ),
      home: _isLoading 
          ? const Scaffold(
              body: Center(
                child: CircularProgressIndicator(),
              ),
            )
          : _isLoggedIn 
              ? const HomeScreen() 
              : const OnboardingScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
