import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import calculate_f1_score

def analyze_results(csv_file='performance_results.csv'):
    # CSV 파일 경로 설정
    csv_path = os.path.join('results', csv_file)
    
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return
    
    # 결과 데이터 로드
    df = pd.read_csv(csv_path)
    
    # F1 스코어 계산
    f1_scores = []
    for index, row in df.iterrows():
        reference = row['reference']  # 참조 응답
        response = row['response']  # 생성된 응답
        _, _, f1 = calculate_f1_score(reference, response)
        f1_scores.append(f1)
    
    df['f1_score'] = f1_scores
    
    # 결과 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    
    # 프레임워크별 성능 통계
    stats = df.groupby(['framework', 'function']).agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'memory_usage': ['mean', 'std', 'min', 'max'],
        'cpu_usage': ['mean', 'std', 'min', 'max'],
        'success': 'mean',
        'f1_score': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("\n=== Performance Comparison: LangChain vs LangGraph ===")
    print("\nDetailed Statistics:")
    print(stats)
    
    # F1 스코어 비교
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='framework', y='f1_score', hue='function')
    plt.title('F1 Score: LangChain vs LangGraph')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig('results/f1_score_comparison.png')
    
    return stats

if __name__ == "__main__":
    analyze_results()