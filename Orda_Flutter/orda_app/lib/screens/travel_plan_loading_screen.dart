import 'package:flutter/material.dart';
import 'dart:async';
import 'travel_plan_detail_screen.dart';

class TravelPlanLoadingScreen extends StatefulWidget {
  final String userName;
  final String personality;
  final String travelStyle;
  final String travelPlan;

  const TravelPlanLoadingScreen({
    super.key,
    required this.userName,
    required this.personality,
    required this.travelStyle,
    required this.travelPlan,
  });

  @override
  State<TravelPlanLoadingScreen> createState() => _TravelPlanLoadingScreenState();
}

class _TravelPlanLoadingScreenState extends State<TravelPlanLoadingScreen> 
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _rotationController;
  late Animation<double> _pulseAnimation;
  late Animation<double> _rotationAnimation;
  
  int _currentStep = 0;
  Timer? _stepTimer;
  
  final List<String> _loadingSteps = [
    '여행 선호도 분석 중...',
    '맞춤형 장소 검색 중...',
    '최적 경로 계산 중...',
    '일정 최종 조정 중...',
    '완성!'
  ];

  @override
  void initState() {
    super.initState();
    
    // 맥박 애니메이션 설정
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(
      begin: 0.9,
      end: 1.1,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));
    
    // 회전 애니메이션 설정
    _rotationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _rotationAnimation = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(CurvedAnimation(
      parent: _rotationController,
      curve: Curves.linear,
    ));
    
    // 애니메이션 시작
    _pulseController.repeat(reverse: true);
    _rotationController.repeat();
    
    // 단계별 로딩 시작
    _startStepAnimation();
    
    // 4초 후 여행일정 상세 페이지로 이동
    Timer(const Duration(seconds: 4), () {
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => TravelPlanDetailScreen(
              userName: widget.userName,
              personality: widget.personality,
              travelStyle: widget.travelStyle,
              travelPlan: widget.travelPlan,
            ),
          ),
        );
      }
    });
  }
  
  void _startStepAnimation() {
    _stepTimer = Timer.periodic(const Duration(milliseconds: 800), (timer) {
      if (_currentStep < _loadingSteps.length - 1) {
        setState(() {
          _currentStep++;
        });
      } else {
        timer.cancel();
      }
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _rotationController.dispose();
    _stepTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Spacer(),
              
              // 사용자 정보 카드
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      widget.personality.contains('에겐') 
                        ? Colors.pink.withOpacity(0.1)
                        : Colors.blue.withOpacity(0.1),
                      widget.personality.contains('에겐')
                        ? Colors.orange.withOpacity(0.1) 
                        : Colors.indigo.withOpacity(0.1),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: widget.personality.contains('에겐') 
                      ? Colors.pink.withOpacity(0.3)
                      : Colors.blue.withOpacity(0.3),
                  ),
                ),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: widget.personality.contains('에겐') 
                              ? Colors.pink.withOpacity(0.2)
                              : Colors.blue.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            widget.personality,
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                              color: widget.personality.contains('에겐') 
                                ? Colors.pink[700]
                                : Colors.blue[700],
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          '${widget.userName}님',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.black87,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Text(
                      widget.travelStyle,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.black87,
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 60),
              
              // 로딩 애니메이션
              AnimatedBuilder(
                animation: _pulseAnimation,
                builder: (context, child) {
                  return Transform.scale(
                    scale: _pulseAnimation.value,
                    child: AnimatedBuilder(
                      animation: _rotationAnimation,
                      builder: (context, child) {
                        return Transform.rotate(
                          angle: _rotationAnimation.value * 2 * 3.14159,
                          child: Container(
                            width: 120,
                            height: 120,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              gradient: LinearGradient(
                                colors: [
                                  widget.personality.contains('에겐') 
                                    ? Colors.pink.withOpacity(0.3)
                                    : Colors.blue.withOpacity(0.3),
                                  widget.personality.contains('에겐')
                                    ? Colors.orange.withOpacity(0.5) 
                                    : Colors.indigo.withOpacity(0.5),
                                ],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: widget.personality.contains('에겐') 
                                    ? Colors.pink.withOpacity(0.2)
                                    : Colors.blue.withOpacity(0.2),
                                  blurRadius: 20,
                                  spreadRadius: 5,
                                ),
                              ],
                            ),
                            child: const Icon(
                              Icons.map_outlined,
                              size: 50,
                              color: Colors.white,
                            ),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
              
              const SizedBox(height: 40),
              
              // 로딩 단계 텍스트
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 500),
                child: Text(
                  _loadingSteps[_currentStep],
                  key: ValueKey(_currentStep),
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: Colors.black87,
                  ),
                ),
              ),
              
              const SizedBox(height: 20),
              
              // 프로그레스 바
              Container(
                width: 200,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(2),
                ),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 500),
                  width: 200 * ((_currentStep + 1) / _loadingSteps.length),
                  height: 4,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        widget.personality.contains('에겐') 
                          ? Colors.pink
                          : Colors.blue,
                        widget.personality.contains('에겐')
                          ? Colors.orange 
                          : Colors.indigo,
                      ],
                    ),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              
              const Spacer(),
              
              // 하단 메시지
              Text(
                '${widget.userName}님만을 위한\n맞춤형 제주 여행일정을 생성하고 있어요',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey[600],
                  height: 1.4,
                ),
              ),
              
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }
}