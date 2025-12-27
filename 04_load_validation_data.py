"""
Validation 데이터 로딩 스크립트
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
VALIDATION_RESOURCE_PATH = '/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Resource'
VALIDATION_LABELING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Labeling'
TRAINING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Training'

def get_training_categories():
    """Training 폴더에 있는 카테고리 목록 반환"""
    categories = set()
    for cat_dir in Path(TRAINING_PATH).iterdir():
        if cat_dir.is_dir() and not cat_dir.name.startswith('.'):
            category_name = cat_dir.name.replace('TS_', '')
            categories.add(category_name)
    return sorted(categories)

def load_validation_data():
    """Validation 데이터 로드"""
    
    # Training에 있는 카테고리만 사용
    training_categories = get_training_categories()
    print(f"Training에 있는 카테고리: {training_categories}")
    
    data = []
    skipped_categories = []
    
    # Validation_Labeling 폴더에서 데이터 로드
    for category_dir in Path(VALIDATION_LABELING_PATH).iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name.replace('VL_', '')
        
        # Training에 없는 카테고리는 스킵
        if category_name not in training_categories:
            skipped_categories.append(category_name)
            print(f"  스킵: {category_name} (Training에 없음)")
            continue
        
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
                    'json_file': str(json_file)
                })
    
    df = pd.DataFrame(data)
    
    print(f"\n총 {len(df)}개의 Validation 데이터 로드됨")
    print(f"스킵된 카테고리: {skipped_categories}")
    
    return df

def analyze_validation_data(df):
    """Validation 데이터 분석"""
    print("\n" + "=" * 50)
    print("Validation 데이터 분석")
    print("=" * 50)
    
    # 카테고리별 분포
    print("\n카테고리별 데이터 개수:")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    # 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    category_counts.plot(kind='bar')
    plt.title('Validation 카테고리별 데이터 분포', fontsize=14, fontweight='bold')
    plt.xlabel('카테고리', fontsize=12)
    plt.ylabel('개수', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    category_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Validation 카테고리별 비율', fontsize=14, fontweight='bold')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('validation_category_distribution.png', dpi=300, bbox_inches='tight')
    print("\n그래프가 'validation_category_distribution.png'로 저장되었습니다.")
    
    # 바운딩 박스 통계
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    print(f"\n바운딩 박스 통계:")
    print(f"평균 너비: {df['bbox_width'].mean():.2f}")
    print(f"평균 높이: {df['bbox_height'].mean():.2f}")
    print(f"평균 면적: {df['bbox_area'].mean():.2f}")
    
    return category_counts

def check_image_files(df):
    """이미지 파일 존재 여부 확인"""
    print("\n" + "=" * 50)
    print("이미지 파일 확인")
    print("=" * 50)
    
    missing_files = []
    found_files = 0
    
    for idx, row in df.iterrows():
        identifier = row['identifier']
        category_name = row['category']
        
        # Validation_Resource 폴더에서 이미지 찾기
        resource_dir = Path(VALIDATION_RESOURCE_PATH) / f"VS_{category_name}"
        image_file = resource_dir / f"{identifier}.png"
        
        if image_file.exists():
            found_files += 1
        else:
            missing_files.append((identifier, category_name))
    
    print(f"\n총 {len(df)}개 중 {found_files}개 이미지 파일 발견")
    if missing_files:
        print(f"누락된 파일: {len(missing_files)}개")
        print("처음 5개:")
        for identifier, category in missing_files[:5]:
            print(f"  {identifier} ({category})")
    else:
        print("모든 이미지 파일이 존재합니다!")
    
    return len(missing_files) == 0

def main():
    """메인 실행 함수"""
    print("Validation 데이터 로딩 시작...")
    
    # 데이터 로드
    df = load_validation_data()
    
    # 데이터 저장
    df.to_csv('validation_data.csv', index=False, encoding='utf-8-sig')
    print("\n데이터가 'validation_data.csv'로 저장되었습니다.")
    
    # 분석
    analyze_validation_data(df)
    
    # 이미지 파일 확인
    all_files_exist = check_image_files(df)
    
    if all_files_exist:
        print("\n✓ 모든 Validation 데이터가 준비되었습니다!")
    else:
        print("\n⚠ 일부 이미지 파일이 누락되었습니다.")
    
    print("\n" + "=" * 50)
    print("Validation 데이터 로딩 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()

