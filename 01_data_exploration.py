"""
피부 질환 이미지 데이터 탐색적 데이터 분석 (EDA) 스크립트
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# 한글 폰트 설정 (Windows - 바탕 폰트 우선)
import matplotlib.font_manager as fm
import matplotlib

def set_korean_font():
    """윈도우 환경에서 바탕 폰트를 우선적으로 사용하는 한글 폰트 설정"""
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 윈도우 환경에서 바탕 폰트 우선, 그 다음 다른 한글 폰트
    preferred_fonts = [
        'Batang',           # 바탕 (영문명)
        'BatangChe',        # 바탕체
        'Malgun Gothic',    # 맑은 고딕
        'Gulim',            # 굴림
        'Dotum',            # 돋움
        'NanumGothic',      # 나눔고딕 (설치된 경우)
        'Noto Sans CJK KR', # Noto Sans (설치된 경우)
    ]
    
    for font_name in preferred_fonts:
        if font_name in font_list:
            plt.rcParams['font.family'] = font_name
            matplotlib.rc('font', family=font_name)
            print(f"[폰트 적용] 한글 폰트: {font_name}")
            break
    else:
        # 위 폰트가 모두 없으면 기본 폰트로
        plt.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rc('font', family='DejaVu Sans')
        print("[경고] 한글 폰트가 적용되지 않을 수 있습니다.")
    
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 데이터 경로
TRAINING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Training'
LABELING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Labeling'

def load_labeling_data():
    """Labeling JSON 파일들을 로드하여 DataFrame으로 변환"""
    data = []
    
    for category_dir in Path(LABELING_PATH).iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name.replace('TL_', '')
        
        for json_file in category_dir.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            for annotation in json_data.get('annotations', []):
                bbox = annotation.get('bbox', {})
                photo = annotation.get('photograph', {})
                diagnosis = annotation.get('diagnosis_info', {})
                
                data.append({
                    'identifier': annotation.get('identifier'),
                    'category': category_name,
                    'file_path': photo.get('file_path'),
                    'image_width': photo.get('width'),
                    'image_height': photo.get('height'),
                    'bbox_x': bbox.get('xpos'),
                    'bbox_y': bbox.get('ypos'),
                    'bbox_width': bbox.get('width'),
                    'bbox_height': bbox.get('height'),
                    'diagnosis_name': diagnosis.get('diagnosis_name'),
                    'race': annotation.get('generated_parameters', {}).get('race'),
                    'json_file': str(json_file)
                })
    
    return pd.DataFrame(data)

def analyze_data_distribution(df):
    """데이터 분포 분석"""
    print("=" * 50)
    print("데이터 분포 분석")
    print("=" * 50)
    
    # 카테고리별 분포
    print("\n카테고리별 데이터 개수:")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    # 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    category_counts.plot(kind='bar')
    plt.title('카테고리별 데이터 분포')
    plt.xlabel('카테고리')
    plt.ylabel('개수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    category_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('카테고리별 비율')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
    print("\n그래프가 'category_distribution.png'로 저장되었습니다.")
    
    return category_counts

def analyze_bbox_statistics(df):
    """바운딩 박스 통계 분석"""
    print("\n" + "=" * 50)
    print("바운딩 박스 통계")
    print("=" * 50)
    
    # 바운딩 박스 크기 통계
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    df['bbox_ratio'] = df['bbox_width'] / df['bbox_height']
    
    print(f"\n바운딩 박스 크기 통계:")
    print(f"평균 너비: {df['bbox_width'].mean():.2f}")
    print(f"평균 높이: {df['bbox_height'].mean():.2f}")
    print(f"평균 면적: {df['bbox_area'].mean():.2f}")
    print(f"평균 비율 (너비/높이): {df['bbox_ratio'].mean():.2f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 바운딩 박스 크기 분포
    axes[0, 0].hist(df['bbox_width'], bins=50, alpha=0.7, label='Width')
    axes[0, 0].hist(df['bbox_height'], bins=50, alpha=0.7, label='Height')
    axes[0, 0].set_xlabel('픽셀')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].set_title('바운딩 박스 크기 분포')
    axes[0, 0].legend()
    
    # 바운딩 박스 면적 분포
    axes[0, 1].hist(df['bbox_area'], bins=50)
    axes[0, 1].set_xlabel('면적 (픽셀²)')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].set_title('바운딩 박스 면적 분포')
    
    # 바운딩 박스 위치 분포
    axes[1, 0].scatter(df['bbox_x'], df['bbox_y'], alpha=0.3, s=1)
    axes[1, 0].set_xlabel('X 위치')
    axes[1, 0].set_ylabel('Y 위치')
    axes[1, 0].set_title('바운딩 박스 위치 분포')
    
    # 카테고리별 바운딩 박스 크기
    category_bbox = df.groupby('category')['bbox_area'].mean().sort_values()
    category_bbox.plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].set_xlabel('평균 면적 (픽셀²)')
    axes[1, 1].set_title('카테고리별 평균 바운딩 박스 크기')
    
    plt.tight_layout()
    plt.savefig('bbox_statistics.png', dpi=300, bbox_inches='tight')
    print("\n그래프가 'bbox_statistics.png'로 저장되었습니다.")

def check_image_files(df):
    """이미지 파일 존재 여부 및 통계 확인"""
    print("\n" + "=" * 50)
    print("이미지 파일 확인")
    print("=" * 50)
    
    # Training 폴더에서 이미지 찾기
    training_images = {}
    for category_dir in Path(TRAINING_PATH).iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name.replace('TS_', '')
        images = list(category_dir.glob('*.png'))
        training_images[category_name] = len(images)
    
    print("\nTraining 폴더 이미지 개수:")
    for category, count in training_images.items():
        print(f"  {category}: {count}개")
    
    # 샘플 이미지 확인
    print("\n샘플 이미지 확인 중...")
    sample_count = 0
    for idx, row in df.iterrows():
        if sample_count >= 5:
            break
        
        # Training 폴더에서 이미지 찾기
        category_name = row['category']
        identifier = row['identifier']
        
        training_dir = Path(TRAINING_PATH) / f"TS_{category_name}"
        image_file = training_dir / f"{identifier}.png"
        
        if image_file.exists():
            img = Image.open(image_file)
            print(f"\n{identifier}:")
            print(f"  크기: {img.size}")
            print(f"  모드: {img.mode}")
            sample_count += 1
    
    return training_images

def generate_summary_report(df):
    """종합 요약 보고서 생성"""
    print("\n" + "=" * 50)
    print("종합 요약 보고서")
    print("=" * 50)
    
    print(f"\n총 데이터 개수: {len(df)}")
    print(f"카테고리 수: {df['category'].nunique()}")
    print(f"고유 식별자 수: {df['identifier'].nunique()}")
    
    print(f"\n이미지 크기:")
    print(f"  너비: {df['image_width'].unique()}")
    print(f"  높이: {df['image_height'].unique()}")
    
    print(f"\n바운딩 박스 통계:")
    print(f"  X 위치 범위: {df['bbox_x'].min()} ~ {df['bbox_x'].max()}")
    print(f"  Y 위치 범위: {df['bbox_y'].min()} ~ {df['bbox_y'].max()}")
    print(f"  너비 범위: {df['bbox_width'].min()} ~ {df['bbox_width'].max()}")
    print(f"  높이 범위: {df['bbox_height'].min()} ~ {df['bbox_height'].max()}")
    
    # 결측치 확인
    print(f"\n결측치 확인:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  결측치 없음")

def main():
    """메인 실행 함수"""
    print("피부 질환 이미지 데이터 탐색적 데이터 분석 시작...")
    
    # 데이터 로드
    print("\n데이터 로딩 중...")
    df = load_labeling_data()
    print(f"총 {len(df)}개의 레이블 데이터를 로드했습니다.")
    
    # 데이터 저장
    df.to_csv('labeling_data.csv', index=False, encoding='utf-8-sig')
    print("데이터가 'labeling_data.csv'로 저장되었습니다.")
    
    # 분석 수행
    analyze_data_distribution(df)
    analyze_bbox_statistics(df)
    check_image_files(df)
    generate_summary_report(df)
    
    print("\n" + "=" * 50)
    print("EDA 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()

