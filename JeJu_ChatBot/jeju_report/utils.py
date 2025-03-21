import time
import psutil
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# 싱글톤 패턴으로 변경
class PerformanceTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.results = []
            cls._instance.csv_file = 'performance_results.csv'
        return cls._instance
    
    def measure_performance(self, framework, reference=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 시작 측정
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_cpu = psutil.cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = str(e)
                    success = False
                
                # 종료 측정
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                end_cpu = psutil.cpu_percent()
                
                # 결과 저장
                self.results.append({
                    'timestamp': datetime.now(),
                    'framework': framework,
                    'function': func.__name__,
                    'execution_time': end_time - start_time,
                    'memory_usage': end_memory - start_memory,
                    'cpu_usage': end_cpu,
                    'success': success,
                    'response': result,
                    'reference': reference  # 참조 응답 추가
                })
                
                # 매 측정마다 CSV 파일 업데이트
                self.save_results()
                
                return result
            return wrapper
        return decorator
    
    def save_results(self):
        if self.results:
            df = pd.DataFrame(self.results)
            os.makedirs('results', exist_ok=True)
            csv_path = os.path.join('results', self.csv_file)
            df.to_csv(csv_path, index=False)
            return df
    
    def clear_results(self):
        self.results = []
        if os.path.exists(os.path.join('results', self.csv_file)):
            os.remove(os.path.join('results', self.csv_file))

# 전역 tracker 인스턴스 생성
performance_tracker = PerformanceTracker()

def calculate_f1_score(reference, generated):
    # 문자열이 아닌 경우 빈 문자열로 처리
    if not isinstance(reference, str):
        reference = ""
    if not isinstance(generated, str):
        generated = ""
    
    # 토큰화
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())
    
    # 빈 토큰 처리
    if not ref_tokens or not gen_tokens:
        return 0.0, 0.0, 0.0
    
    # Binarize the tokens
    mlb = MultiLabelBinarizer()
    ref_binarized = mlb.fit_transform([ref_tokens])
    gen_binarized = mlb.transform([gen_tokens])
    
    # Precision, Recall, F1 Score
    precision = precision_score(ref_binarized, gen_binarized, average='micro', zero_division=0)
    recall = recall_score(ref_binarized, gen_binarized, average='micro', zero_division=0)
    f1 = f1_score(ref_binarized, gen_binarized, average='micro', zero_division=0)
    
    return precision, recall, f1